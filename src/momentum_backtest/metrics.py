"""Performance statistics for a monthly return series.

Nothing here is exotic. The goal is an honest summary a reader can sanity
check: how much it made, how bumpy the ride was, and how deep the worst hole
got. All annualization assumes monthly data (12 periods per year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate implied by the return stream."""
    if returns.empty:
        return float("nan")
    growth = (1.0 + returns).prod()
    years = len(returns) / PERIODS_PER_YEAR
    if years <= 0 or growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of returns."""
    return returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio.

    ``risk_free_rate`` is annualized and converted to a monthly figure before
    being subtracted from each period's return.
    """
    rf_monthly = (1.0 + risk_free_rate) ** (1.0 / PERIODS_PER_YEAR) - 1.0
    excess = returns - rf_monthly
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return (excess.mean() / vol) * np.sqrt(PERIODS_PER_YEAR)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation in the denominator)."""
    rf_monthly = (1.0 + risk_free_rate) ** (1.0 / PERIODS_PER_YEAR) - 1.0
    excess = returns - rf_monthly
    downside = excess[excess < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) else np.nan
    if not downside_dev or np.isnan(downside_dev):
        return float("nan")
    return (excess.mean() / downside_dev) * np.sqrt(PERIODS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve (a negative number)."""
    equity = (1.0 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return drawdown.min()


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by the magnitude of the max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr(returns) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    """Share of months with a positive return."""
    if returns.empty:
        return float("nan")
    return (returns > 0).mean()


def summarize(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    """Bundle the headline statistics into a single dictionary."""
    stats = {
        "Total return": (1.0 + returns).prod() - 1.0,
        "CAGR": cagr(returns),
        "Annualized volatility": annualized_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns, risk_free_rate),
        "Sortino ratio": sortino_ratio(returns, risk_free_rate),
        "Max drawdown": max_drawdown(returns),
        "Calmar ratio": calmar_ratio(returns),
        "Hit rate": hit_rate(returns),
        "Months": float(len(returns)),
    }
    if turnover is not None and not turnover.empty:
        # Average one-way turnover per month, annualized for readability.
        stats["Avg annual turnover"] = turnover.mean() * PERIODS_PER_YEAR
    return stats


def format_summary(stats: dict[str, float]) -> str:
    """Render the summary dictionary as an aligned, readable block of text."""
    percent_keys = {
        "Total return",
        "CAGR",
        "Annualized volatility",
        "Max drawdown",
        "Hit rate",
        "Avg annual turnover",
    }
    lines = []
    width = max(len(k) for k in stats)
    for key, value in stats.items():
        if key in percent_keys:
            shown = f"{value * 100:,.2f}%"
        elif key == "Months":
            shown = f"{value:,.0f}"
        else:
            shown = f"{value:,.2f}"
        lines.append(f"{key:<{width}}  {shown}")
    return "\n".join(lines)
