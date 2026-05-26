"""Benchmark-relative performance: is the strategy actually adding anything?

A strategy's absolute return is only half the question. The other half is
whether it beat a simple benchmark, and how reliably. These are the standard
active-management measures:

    active return    strategy return minus benchmark return, period by period
    tracking error   annualized volatility of the active return
    information ratio annualized active return divided by tracking error, i.e.
                      reward per unit of deviation from the benchmark
    beta             sensitivity of the strategy to the benchmark

A high absolute return that simply tracks the benchmark has a low information
ratio, which is the honest verdict: it was beta, not skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12   # monthly strategy


def _align(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    df = pd.concat([returns, benchmark], axis=1, keys=["r", "b"]).dropna()
    return df


def tracking_error(returns: pd.Series, benchmark: pd.Series) -> float:
    df = _align(returns, benchmark)
    active = df["r"] - df["b"]
    return float(active.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    df = _align(returns, benchmark)
    active = df["r"] - df["b"]
    sd = active.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((active.mean() / sd) * np.sqrt(PERIODS_PER_YEAR))


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    df = _align(returns, benchmark)
    var_b = df["b"].var(ddof=1)
    if var_b == 0 or np.isnan(var_b):
        return float("nan")
    return float(df["r"].cov(df["b"]) / var_b)


def summarize(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    return {
        "Tracking error": tracking_error(returns, benchmark),
        "Information ratio": information_ratio(returns, benchmark),
        "Beta": beta(returns, benchmark),
    }
