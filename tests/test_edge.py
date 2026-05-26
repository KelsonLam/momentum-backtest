"""Edge-case and validation tests for the backtest engine."""

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


def _months(n):
    return pd.date_range("2010-01-31", periods=n, freq="ME")


def test_config_rejects_bad_quantile():
    with pytest.raises(ValueError):
        BacktestConfig(top_quantile=1.5)
    with pytest.raises(ValueError):
        BacktestConfig(top_quantile=0.0)


def test_config_rejects_lookback_not_above_gap():
    with pytest.raises(ValueError):
        BacktestConfig(lookback_months=3, gap_months=3)


def test_config_rejects_negative_cost():
    with pytest.raises(ValueError):
        BacktestConfig(transaction_cost_bps=-1)


def test_run_rejects_insufficient_history():
    prices = pd.DataFrame({"A": np.linspace(100, 110, 6)}, index=_months(6))
    with pytest.raises(ValueError):
        run_backtest(prices, BacktestConfig(lookback_months=12, gap_months=1))


def test_run_rejects_empty_frame():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame(), BacktestConfig())


def test_full_quantile_holds_everything():
    prices = pd.DataFrame(
        {"A": np.linspace(100, 200, 20), "B": np.linspace(100, 150, 20)},
        index=_months(20),
    )
    res = run_backtest(prices, BacktestConfig(lookback_months=6, gap_months=1, top_quantile=1.0))
    active = res.weights.loc[res.returns.index[-1]]
    assert (active > 0).sum() == 2          # both names held when quantile is 1
