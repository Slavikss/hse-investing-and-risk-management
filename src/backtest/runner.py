"""Expanding-window VaR vs следующий день P&L; Kupiec/Christoffersen."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.risk.var import compute_var_timeseries
from src.backtest.kupiec import run_all_kupiec_tests, kupiec_results_to_df, KupiecResult

logger = logging.getLogger(__name__)

BACKTEST_PERIODS = {
    "COVID_2020":    ("2019-01-01", "2020-03-23"),
    "RateHike_2022": ("2019-01-01", "2022-12-31"),
}

TRAIN_MIN = 252


@dataclass
class BacktestRun:
    period_name:  str
    start_train:  str
    end_test:     str
    portfolio_pnl:  pd.Series
    var_series:     dict[str, pd.Series]
    exceedances:    dict[str, pd.Series]
    kupiec_results: list[KupiecResult]
    kupiec_table:   pd.DataFrame


def run_backtest(
    asset_returns_usd: pd.DataFrame,
    weights:           dict[str, float],
    period:            str = "COVID_2020",
    conf:              float = 0.95,
    methods:           list[str] | None = None,
    n_sim:             int = 2000,
) -> BacktestRun:
    """Бэктест по ключу period из BACKTEST_PERIODS."""
    if methods is None:
        methods = ["historical", "parametric", "mc_normal", "mc_t"]

    start_train, end_test = BACKTEST_PERIODS[period]

    mask = (asset_returns_usd.index >= start_train) & (asset_returns_usd.index <= end_test)
    rets = asset_returns_usd[mask].copy()

    tickers = [t for t in weights if t in rets.columns]
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()
    w_dict = dict(zip(tickers, w.tolist()))

    rets_clean = rets[tickers].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    port_pnl = pd.Series(
        rets_clean.values @ w,
        index=rets.index,
        name="port_pnl",
    )

    logger.info("Backtest %s: %d days, training min=%d", period, len(rets), TRAIN_MIN)

    var_series: dict[str, pd.Series] = {}
    for method in methods:
        logger.info("  Computing %s VaR timeseries...", method)
        vs = compute_var_timeseries(
            rets, w_dict, conf=conf, method=method, window=TRAIN_MIN  # type: ignore[arg-type]
        )
        var_series[method] = vs

    exceedances: dict[str, pd.Series] = {}
    for method, vs in var_series.items():
        pnl_aligned = port_pnl.reindex(vs.index)
        exceed = ((-pnl_aligned) > vs).astype(int)
        exceedances[method] = exceed.rename(f"exceed_{method}")

    valid_dates = var_series[methods[0]].index
    for m in methods[1:]:
        valid_dates = valid_dates.intersection(var_series[m].index)

    pnl_aligned = port_pnl.reindex(valid_dates)
    pnl_arr = pnl_aligned.values.astype(float)
    var_dict_arr = {
        m: var_series[m].reindex(valid_dates).astype(float).values
        for m in methods
        if m in var_series
    }

    kupiec_results = run_all_kupiec_tests(pnl_arr, var_dict_arr, conf)
    kupiec_table   = kupiec_results_to_df(kupiec_results)

    return BacktestRun(
        period_name=period,
        start_train=start_train,
        end_test=end_test,
        portfolio_pnl=port_pnl,
        var_series=var_series,
        exceedances=exceedances,
        kupiec_results=kupiec_results,
        kupiec_table=kupiec_table,
    )


def run_all_backtests(
    asset_returns_usd: pd.DataFrame,
    weights:           dict[str, float],
    conf:              float = 0.95,
) -> dict[str, BacktestRun]:
    """Запускает бэктесты для всех периодов."""
    results = {}
    for period in BACKTEST_PERIODS:
        logger.info("Backtest period: %s", period)
        try:
            results[period] = run_backtest(asset_returns_usd, weights, period, conf)
        except Exception as exc:
            logger.error("Backtest %s failed: %s", period, exc)
    return results
