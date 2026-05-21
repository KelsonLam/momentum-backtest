"""Portfolio construction and the return calculation.

The flow is deliberately boring, which is the point in a backtest:

    1. Score every asset each month (see signals.py).
    2. At each month end, go long an equal-weighted basket of the highest
       scoring names. Optionally short the lowest scoring names.
    3. Hold those weights through the following month and book the return.
    4. Charge a transaction cost proportional to how much the book turned over.

Two details matter for trusting the result:

    * No lookahead. Weights decided at the end of month t are multiplied by the
      return of month t+1. We never use a return to pick the position that
      earns it.
    * Costs are explicit. The headline number is net of an estimated trading
      cost, not a frictionless fantasy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import momentum_score


@dataclass
class BacktestConfig:
    lookback_months: int = 12
    gap_months: int = 1
    top_quantile: float = 0.30
    long_short: bool = False
    transaction_cost_bps: float = 10.0
    risk_free_rate: float = 0.0   # annualized, used downstream by the metrics

    def __post_init__(self) -> None:
        # Fail loudly and early on nonsense settings, rather than producing a
        # silently wrong backtest.
        if self.lookback_months <= self.gap_months:
            raise ValueError(
                "lookback_months must be greater than gap_months "
                f"(got lookback={self.lookback_months}, gap={self.gap_months})."
            )
        if self.gap_months < 0:
            raise ValueError("gap_months cannot be negative.")
        if not 0.0 < self.top_quantile <= 1.0:
            raise ValueError("top_quantile must be in the interval (0, 1].")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative.")


@dataclass
class BacktestResult:
    returns: pd.Series          # net monthly returns of the strategy
    gross_returns: pd.Series    # before transaction costs
    equity_curve: pd.Series     # growth of one unit, net of costs
    weights: pd.DataFrame       # target weight per asset per month
    turnover: pd.Series         # fraction of the book traded each month
    config: BacktestConfig


def _target_weights(
    scores: pd.DataFrame, top_quantile: float, long_short: bool
) -> pd.DataFrame:
    """Build the month-by-month target weight matrix from the score matrix.

    Each row holds an equal-weighted long book (gross long exposure of 1.0).
    When ``long_short`` is on, an equal-weighted short book of the same size is
    added, giving a roughly dollar-neutral portfolio.
    """
    weights = pd.DataFrame(
        0.0, index=scores.index, columns=scores.columns
    )

    for date, row in scores.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n == 0:
            continue

        k = max(1, int(round(top_quantile * n)))
        # Guard against the long and short books overlapping in a tiny universe.
        if long_short:
            k = min(k, n // 2 if n >= 2 else 1)
            if k == 0:
                continue

        ranked = valid.sort_values(ascending=False)
        longs = ranked.index[:k]
        weights.loc[date, longs] = 1.0 / k

        if long_short:
            shorts = ranked.index[-k:]
            weights.loc[date, shorts] = -1.0 / k

    return weights


def run_backtest(
    monthly_prices: pd.DataFrame, config: BacktestConfig
) -> BacktestResult:
    """Run the momentum backtest on a frame of month-end prices."""
    if monthly_prices.empty or monthly_prices.shape[1] == 0:
        raise ValueError("monthly_prices is empty. Nothing to backtest.")

    needed = config.lookback_months + config.gap_months + 1
    if len(monthly_prices) < needed:
        raise ValueError(
            f"Not enough history: the formation window needs at least {needed} "
            f"months but only {len(monthly_prices)} were given. Use a longer "
            "date range or a shorter lookback."
        )

    scores = momentum_score(
        monthly_prices,
        lookback_months=config.lookback_months,
        gap_months=config.gap_months,
    )
    monthly_returns = monthly_prices.pct_change()

    weights = _target_weights(scores, config.top_quantile, config.long_short)

    # Weights set at the end of month t earn the return of month t+1, so the
    # weights are shifted forward by one period before meeting the returns.
    held_weights = weights.shift(1)
    gross_returns = (held_weights * monthly_returns).sum(axis=1)

    # Turnover is how much of the book we trade to reach the new target. The
    # cost of trading at the end of month t is borne over the t+1 holding
    # period, so it is shifted forward in lockstep with the weights above.
    turnover = weights.diff().abs().sum(axis=1)
    cost_rate = config.transaction_cost_bps / 10_000.0
    costs = turnover.shift(1) * cost_rate

    net_returns = (gross_returns - costs).fillna(0.0)

    # Trim the warm-up months where there was no position yet.
    first_trade = weights.abs().sum(axis=1)
    active_from = first_trade[first_trade > 0].index.min()
    if active_from is not None:
        net_returns = net_returns.loc[active_from:]
        gross_returns = gross_returns.loc[active_from:]
        turnover = turnover.loc[active_from:]

    equity_curve = (1.0 + net_returns).cumprod()

    return BacktestResult(
        returns=net_returns,
        gross_returns=gross_returns,
        equity_curve=equity_curve,
        weights=weights,
        turnover=turnover,
        config=config,
    )
