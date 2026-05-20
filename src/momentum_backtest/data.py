"""Price loading behind a small, vendor-agnostic interface.

The strategy code only ever asks for a tidy DataFrame of adjusted close prices
(dates on the index, tickers on the columns). It never talks to a data vendor
directly. That separation means you can switch from yfinance to a paid feed
like Tiingo or Polygon by writing one new loader, with nothing else changing.

Downloads are cached to local parquet files keyed by the request, so a repeated
run is instant and works offline once the cache is warm.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


class PriceLoader(Protocol):
    """Anything that can hand back adjusted close prices.

    Implement this to plug in a different data vendor. The only contract is the
    shape of the returned frame: a DatetimeIndex and one column per ticker.
    """

    def load(
        self, tickers: Sequence[str], start: str, end: str
    ) -> pd.DataFrame: ...


def _cache_key(tickers: Sequence[str], start: str, end: str) -> str:
    """A short, stable filename for one specific request."""
    raw = "|".join([",".join(sorted(tickers)), start, end])
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"prices_{digest}.parquet"


class YFinanceLoader:
    """Adjusted close prices from Yahoo Finance via the yfinance package.

    Results are cached to ``cache_dir``. Pass ``use_cache=False`` to force a
    fresh download (handy when you suspect the cache is stale).
    """

    def __init__(
        self, cache_dir: Path | str = DEFAULT_CACHE_DIR, use_cache: bool = True
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    def load(
        self, tickers: Sequence[str], start: str, end: str
    ) -> pd.DataFrame:
        tickers = list(tickers)
        cache_path = self.cache_dir / _cache_key(tickers, start, end)

        if self.use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        prices = self._download(tickers, start, end)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        return prices

    @staticmethod
    def _download(
        tickers: Sequence[str], start: str, end: str
    ) -> pd.DataFrame:
        # Imported lazily so the rest of the package (and the test suite) does
        # not need yfinance installed just to import this module.
        import yfinance as yf

        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,   # "Close" is already split- and dividend-adjusted
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                "No data returned. Check the tickers and the date range."
            )

        # With auto_adjust=True the adjusted price lives in the "Close" field.
        # yfinance returns a column MultiIndex of (field, ticker) for more than
        # one ticker and a flat frame for a single ticker, so handle both.
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            prices.columns = [tickers[0]]

        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        # Keep a column only if it has at least one real observation.
        prices = prices.dropna(axis=1, how="all")
        return prices


def to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to the last observed price each month.

    Momentum is a monthly strategy, so we work on month-end prices throughout.
    """
    monthly = prices.resample("ME").last()
    return monthly
