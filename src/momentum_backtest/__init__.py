"""A from-scratch cross-sectional momentum backtest.

The package is split into small, focused modules so each piece can be read,
tested, and swapped on its own:

    data      loading prices (with caching) behind a vendor-agnostic interface
    signals   turning prices into a momentum score
    backtest  portfolio construction and the return calculation
    metrics   performance statistics
    plotting  equity curve and drawdown charts
"""

__version__ = "0.1.0"
