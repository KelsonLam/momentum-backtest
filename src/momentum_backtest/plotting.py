"""Two simple charts: the equity curve and the drawdown.

Matplotlib only, so there are no surprise dependencies. Each function returns
the Matplotlib Figure so a caller can show it, save it, or tweak it further.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Momentum strategy equity curve",
):
    """Plot growth of one unit, optionally against a benchmark."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve.index, equity_curve.values, label="Momentum (net)")
    if benchmark is not None:
        ax.plot(
            benchmark.index, benchmark.values,
            label="Benchmark", alpha=0.7,
        )
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 unit")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(
    returns: pd.Series, title: str = "Drawdown"
):
    """Plot the underwater curve (drawdown from the running peak)."""
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(
        drawdown.index, drawdown.values, 0.0, color="tab:red", alpha=0.4
    )
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path | str) -> Path:
    """Save a figure to disk, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path
