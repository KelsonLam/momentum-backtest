"""Turning prices into a momentum score.

The score we use is the classic "12 minus 1" momentum: the cumulative return
measured from twelve months ago up to one month ago. The most recent month is
deliberately skipped because, at the one-month horizon, returns tend to reverse
rather than persist, and including it muddies the signal.
"""

from __future__ import annotations

import pandas as pd


def momentum_score(
    monthly_prices: pd.DataFrame,
    lookback_months: int = 12,
    gap_months: int = 1,
) -> pd.DataFrame:
    """Compute the momentum score for every asset at every month end.

    The score at month ``t`` is::

        price[t - gap] / price[t - lookback] - 1

    With the defaults (lookback 12, gap 1) that is the return from twelve
    months ago to one month ago, i.e. the standard 12-1 momentum window.

    Crucially, the score at ``t`` only uses prices up to and including ``t``,
    so a portfolio formed at ``t`` and held over the next month never peeks at
    future data.
    """
    if lookback_months <= gap_months:
        raise ValueError("lookback_months must be larger than gap_months.")

    recent = monthly_prices.shift(gap_months)
    base = monthly_prices.shift(lookback_months)
    score = recent / base - 1.0
    return score
