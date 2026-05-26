"""Tests for the benchmark-relative metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from momentum_backtest import benchmark


def _idx(n):
    return pd.date_range("2015-01-31", periods=n, freq="ME")


def test_identical_series_have_zero_tracking_error():
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.03, 60), index=_idx(60))
    assert benchmark.tracking_error(r, r) == pytest.approx(0.0, abs=1e-12)


def test_beta_against_self_is_one():
    r = pd.Series(np.random.default_rng(1).normal(0.01, 0.03, 60), index=_idx(60))
    assert benchmark.beta(r, r) == pytest.approx(1.0)


def test_consistent_outperformance_gives_positive_ir():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0.005, 0.02, 120), index=_idx(120))
    r = b + 0.004                      # a steady, low-noise edge each month
    assert benchmark.information_ratio(r, b) > 0
    assert benchmark.tracking_error(r, b) == pytest.approx(0.0, abs=1e-9)


def test_beta_scales_with_exposure():
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(0.0, 0.04, 200), index=_idx(200))
    r = 1.5 * b                        # twice-and-a-half... here 1.5x the benchmark
    assert benchmark.beta(r, b) == pytest.approx(1.5, abs=1e-9)
