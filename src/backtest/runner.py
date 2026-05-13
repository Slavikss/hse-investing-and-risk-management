"""Expanding-window VaR vs следующий день P&L; Kupiec/Christoffersen."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.kupiec import KupiecResult, kupiec_results_to_df, run_all_kupiec_tests
from src.risk.var import compute_var_timeseries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestPeriod:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    regime: str


BACKTEST_PERIODS: dict[str, BacktestPeriod] = {
    "COVID_2020": BacktestPeriod(
        train_start="2019-01-01",
        train_end="2019-12-31",
        test_start="2020-01-01",
        test_end="2020-12-31",
        regime="crisis",
    ),
    "Calm_2021": BacktestPeriod(
        train_start="2019-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-12-31",
        regime="normal",
    ),
    "RateHike_2022": BacktestPeriod(
        train_start="2019-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        regime="crisis",
    ),
}

REGIME_HYPERPARAMS: dict[str, dict[str, float | int]] = {
    "normal": {"lambda_ewma": 0.97, "n_sim": 2000},
    "crisis": {"lambda_ewma": 0.90, "n_sim": 4000},
}


@dataclass
class BacktestRun:
    period_name: str
    regime: str
    start_train: str
    start_test: str
    end_test: str
    ewma_lambda: float
    n_sim: int
    portfolio_pnl: pd.Series
    var_series: dict[str, pd.Series]
    exceedances: dict[str, pd.Series]
    kupiec_results: list[KupiecResult]
    kupiec_table: pd.DataFrame


def run_backtest(
    asset_returns_usd: pd.DataFrame,
    weights: dict[str, float],
    period: str = "COVID_2020",
    conf: float = 0.95,
    methods: list[str] | None = None,
    n_sim: int = 2000,
    regime_hyperparams: dict[str, dict[str, float | int]] | None = None,
) -> BacktestRun:
    """Бэктест по ключу period из BACKTEST_PERIODS."""
    if methods is None:
        methods = ["historical", "parametric", "mc_normal", "mc_t"]
    if period not in BACKTEST_PERIODS:
        raise KeyError(f"Unknown period '{period}'. Available: {list(BACKTEST_PERIODS)}")

    p = BACKTEST_PERIODS[period]
    params = regime_hyperparams or REGIME_HYPERPARAMS
    regime_params = params.get(p.regime, {})
    lam = float(regime_params.get("lambda_ewma", 0.94))
    n_sim_eff = int(regime_params.get("n_sim", n_sim))

    mask = (asset_returns_usd.index >= p.train_start) & (asset_returns_usd.index <= p.test_end)
    rets = asset_returns_usd[mask].copy()
    if rets.empty:
        raise ValueError(f"No data for period '{period}' in [{p.train_start}, {p.test_end}]")

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

    train_obs = int((rets.index <= pd.Timestamp(p.train_end)).sum())
    if train_obs < 60:
        raise ValueError(
            f"Too few train observations for period '{period}': {train_obs} (need at least 60)."
        )
    test_mask = (rets.index >= pd.Timestamp(p.test_start)) & (rets.index <= pd.Timestamp(p.test_end))
    test_obs = int(test_mask.sum())
    if test_obs < 30:
        raise ValueError(
            f"Too few test observations for period '{period}': {test_obs} (need at least 30)."
        )

    logger.info(
        "Backtest %s (%s): total=%d train=%d test=%d lambda=%.2f n_sim=%d",
        period, p.regime, len(rets), train_obs, test_obs, lam, n_sim_eff
    )

    var_series: dict[str, pd.Series] = {}
    for method in methods:
        logger.info("  Computing %s VaR timeseries...", method)
        vs = compute_var_timeseries(
            rets,
            w_dict,
            conf=conf,
            method=method,
            window=train_obs,  # type: ignore[arg-type]
            n_sim=n_sim_eff,
            lam=lam,
        )
        vs = vs[(vs.index >= pd.Timestamp(p.test_start)) & (vs.index <= pd.Timestamp(p.test_end))]
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
    kupiec_table = kupiec_results_to_df(kupiec_results)

    return BacktestRun(
        period_name=period,
        regime=p.regime,
        start_train=p.train_start,
        start_test=p.test_start,
        end_test=p.test_end,
        ewma_lambda=lam,
        n_sim=n_sim_eff,
        portfolio_pnl=port_pnl,
        var_series=var_series,
        exceedances=exceedances,
        kupiec_results=kupiec_results,
        kupiec_table=kupiec_table,
    )


def run_all_backtests(
    asset_returns_usd: pd.DataFrame,
    weights: dict[str, float],
    conf: float = 0.95,
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
