"""Command line entry point for the momentum backtest.

Examples
--------
Run with everything from config.yaml::

    python scripts/run_backtest.py

Override a few knobs without touching the file::

    python scripts/run_backtest.py --lookback 6 --top-quantile 0.2 --long-short

Save the charts to the results folder::

    python scripts/run_backtest.py --save-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make the package importable when run straight from a clone, with no install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from momentum_backtest.backtest import BacktestConfig, run_backtest
from momentum_backtest.data import YFinanceLoader, to_month_end
from momentum_backtest.metrics import format_summary, summarize
from momentum_backtest import plotting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the momentum backtest.")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="Path to the YAML config file.",
    )
    p.add_argument("--start", help="Override the start date (YYYY-MM-DD).")
    p.add_argument("--end", help="Override the end date (YYYY-MM-DD).")
    p.add_argument("--lookback", type=int, help="Formation window in months.")
    p.add_argument("--gap", type=int, help="Months skipped before formation end.")
    p.add_argument("--top-quantile", type=float, help="Long book size, 0 to 1.")
    p.add_argument(
        "--long-short", action="store_true",
        help="Short the bottom quantile as well as going long the top.",
    )
    p.add_argument("--cost-bps", type=float, help="Transaction cost in bps.")
    p.add_argument(
        "--no-cache", action="store_true",
        help="Force a fresh download instead of using cached prices.",
    )
    p.add_argument(
        "--save-plots", action="store_true",
        help="Write the equity curve and drawdown charts to results/.",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    start = args.start or cfg["period"]["start"]
    end = args.end or cfg["period"]["end"]
    universe = cfg["universe"]

    strat = cfg["strategy"]
    backtest_cfg = BacktestConfig(
        lookback_months=args.lookback or strat["lookback_months"],
        gap_months=args.gap if args.gap is not None else strat["gap_months"],
        top_quantile=args.top_quantile or strat["top_quantile"],
        long_short=args.long_short or strat["long_short"],
        transaction_cost_bps=(
            args.cost_bps
            if args.cost_bps is not None
            else cfg["costs"]["transaction_cost_bps"]
        ),
        risk_free_rate=cfg.get("risk_free_rate", 0.0),
    )

    print(f"Loading prices for {len(universe)} tickers, {start} to {end} ...")
    loader = YFinanceLoader(use_cache=not args.no_cache)
    prices = loader.load(universe, start, end)
    monthly = to_month_end(prices)

    print("Running backtest ...")
    result = run_backtest(monthly, backtest_cfg)

    stats = summarize(
        result.returns,
        risk_free_rate=backtest_cfg.risk_free_rate,
        turnover=result.turnover,
    )

    print("\nStrategy performance (net of costs)")
    print("-" * 38)
    print(format_summary(stats))

    if args.save_plots:
        # Equal-weight buy and hold of the same universe, as a yardstick.
        monthly_returns = monthly.pct_change()
        bench_returns = monthly_returns.loc[result.returns.index].mean(axis=1)
        benchmark = (1.0 + bench_returns.fillna(0.0)).cumprod()

        fig1 = plotting.plot_equity_curve(result.equity_curve, benchmark)
        fig2 = plotting.plot_drawdown(result.returns)
        out1 = plotting.save_figure(fig1, "results/equity_curve.png")
        out2 = plotting.save_figure(fig2, "results/drawdown.png")
        print(f"\nSaved charts to {out1} and {out2}")


if __name__ == "__main__":
    main()
