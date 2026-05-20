"""Tests for the backtest engine and metrics.

These use synthetic price paths so they run fast and offline, with no
dependence on a data vendor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from momentum_backtest.backtest import BacktestConfig, run_backtest
from momentum_backtest.signals import momentum_score
from momentum_backtest import metrics


def _month_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-31", periods=n, freq="ME")


def test_momentum_score_matches_definition():
    idx = _month_index(15)
    prices = pd.DataFrame({"A": np.linspace(100, 240, 15)}, index=idx)
    score = momentum_score(prices, lookback_months=12, gap_months=1)
    # At the last date: price one month ago / price twelve months ago - 1.
    expected = prices["A"].iloc[-2] / prices["A"].iloc[-13] - 1.0
    assert score["A"].iloc[-1] == pytest.approx(expected)


def test_winner_gets_the_weight():
    # A rises steadily, B falls steadily. Momentum should hold A, not B.
    idx = _month_index(24)
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 200, 24),
            "B": np.linspace(200, 100, 24),
        },
        index=idx,
    )
    cfg = BacktestConfig(top_quantile=0.5, long_short=False)
    result = run_backtest(prices, cfg)
    last_weights = result.weights.loc[result.returns.index[-1]]
    assert last_weights["A"] > 0
    assert last_weights["B"] == 0


def test_no_lookahead_uses_lagged_weights():
    # Hand-built case: the engine must earn month t+1 with weights set at t.
    idx = _month_index(20)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.01, 0.04, size=(20, 3)), axis=0),
        index=idx,
        columns=["A", "B", "C"],
    )
    cfg = BacktestConfig(lookback_months=6, gap_months=1, top_quantile=0.34)
    result = run_backtest(prices, cfg)

    monthly_returns = prices.pct_change()
    expected = (result.weights.shift(1) * monthly_returns).sum(axis=1)
    expected = expected.loc[result.gross_returns.index]
    pd.testing.assert_series_equal(
        result.gross_returns, expected, check_names=False
    )


def test_long_short_is_dollar_neutral():
    idx = _month_index(24)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.005, 0.05, size=(24, 6)), axis=0),
        index=idx,
        columns=list("ABCDEF"),
    )
    cfg = BacktestConfig(top_quantile=0.33, long_short=True)
    result = run_backtest(prices, cfg)
    active = result.weights.loc[result.returns.index]
    nonzero = active[active.abs().sum(axis=1) > 0]
    # Each active row should net to roughly zero dollars.
    assert np.allclose(nonzero.sum(axis=1), 0.0, atol=1e-9)


def test_costs_reduce_returns():
    idx = _month_index(24)
    rng = np.random.default_rng(2)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.01, 0.04, size=(24, 5)), axis=0),
        index=idx,
        columns=list("ABCDE"),
    )
    free = run_backtest(prices, BacktestConfig(transaction_cost_bps=0.0))
    costly = run_backtest(prices, BacktestConfig(transaction_cost_bps=50.0))
    assert costly.returns.sum() < free.returns.sum()


def test_cagr_of_known_series():
    # Twelve months of exactly 1% should compound to (1.01**12 - 1) per year.
    returns = pd.Series([0.01] * 12, index=_month_index(12))
    assert metrics.cagr(returns) == pytest.approx(1.01 ** 12 - 1.0)


def test_max_drawdown_is_negative_or_zero():
    returns = pd.Series([0.05, -0.10, 0.02, -0.03], index=_month_index(4))
    assert metrics.max_drawdown(returns) <= 0.0
