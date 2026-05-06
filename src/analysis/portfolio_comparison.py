"""Сравнение портфелей и бенчмарков по метрикам риска/доходности."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.risk.var import compute_all_var, ewma_volatility, portfolio_returns_usd

logger = logging.getLogger(__name__)

PORTFOLIO_VARIANTS: dict[str, dict[str, float]] = {
    "Default":  {"SBER": 0.15, "LKOH": 0.15, "GAZP": 0.10, "NVDA": 0.20, "MSFT": 0.20, "AAPL": 0.20},
    "Equal":    {"SBER": 1/6,  "LKOH": 1/6,  "GAZP": 1/6,  "NVDA": 1/6,  "MSFT": 1/6,  "AAPL": 1/6},
    "RU-heavy": {"SBER": 0.25, "LKOH": 0.25, "GAZP": 0.20, "NVDA": 0.10, "MSFT": 0.10, "AAPL": 0.10},
    "US-heavy": {"SBER": 0.05, "LKOH": 0.05, "GAZP": 0.05, "NVDA": 0.30, "MSFT": 0.30, "AAPL": 0.25},
}


@dataclass
class PortfolioMetrics:
    name:          str
    weights:       dict[str, float]
    ann_return:    float
    ann_vol:       float
    sharpe:        float
    max_drawdown:  float
    var_95_1d:     float
    es_95_1d:      float
    var_99_1d:     float
    var_95_10d:    float
    calmar:        float
    beta_spy:      Optional[float] = None
    beta_imoex:    Optional[float] = None
    info_ratio:    Optional[float] = None

    def to_series(self) -> pd.Series:
        def _f(x: float | None) -> str:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "—"
            return f"{x:.4f}"

        return pd.Series({
            "Портфель":         self.name,
            "Ann. Return":      f"{self.ann_return:.2%}",
            "Ann. Vol":         f"{self.ann_vol:.2%}",
            "Sharpe":           _f(self.sharpe),
            "Calmar":           _f(self.calmar),
            "Max DD":           f"{self.max_drawdown:.2%}",
            "VaR 95% 1d":       f"{self.var_95_1d:.3%}",
            "ES 95% 1d":        f"{self.es_95_1d:.3%}",
            "VaR 99% 1d":       f"{self.var_99_1d:.3%}",
            "VaR 95% 10d":      f"{self.var_95_10d:.3%}",
            "β(SPY)":           _f(self.beta_spy),
            "β(IMOEX)":         _f(self.beta_imoex),
        })


def _max_drawdown(port_ret: np.ndarray) -> float:
    cum = np.exp(np.cumsum(port_ret))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1.0)
    return float(dd.min())


def _ols_beta(port_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    y, x = port_ret.align(bench_ret, join="inner")
    y, x = y.dropna(), x.dropna()
    common = y.index.intersection(x.index)
    y, x = y.loc[common], x.loc[common]
    if len(y) < 30:
        return None
    X = np.column_stack([np.ones(len(x)), x.values])
    coef, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    return float(coef[1])


def compute_metrics(
    name: str,
    weights: dict[str, float],
    returns_usd: pd.DataFrame,
    bench_spy: Optional[pd.Series] = None,
    bench_imoex: Optional[pd.Series] = None,
    rf_daily: float = 0.0,
    conf: float = 0.95,
    n_sim: int = 1000,
) -> PortfolioMetrics:
    valid = {t: w for t, w in weights.items() if t in returns_usd.columns}
    if not valid:
        raise ValueError(f"None of {list(weights)} found in returns columns")
    w_arr = np.array(list(valid.values()), dtype=float)
    w_arr /= w_arr.sum()
    valid = dict(zip(valid.keys(), w_arr.tolist()))

    # build clean sub-DataFrame and keep its own index (length may differ from returns_usd
    # because portfolio_returns_usd drops all-NaN rows internally)
    sub = (
        returns_usd[list(valid)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
        .fillna(0.0)
    )
    sub_w = np.array([valid[t] for t in sub.columns], dtype=float)
    sub_w /= sub_w.sum()
    port_ret: pd.Series = pd.Series(sub.values @ sub_w, index=sub.index)

    ann_return = float(port_ret.mean() * 252)
    ann_vol    = float(port_ret.std()  * np.sqrt(252))
    sharpe     = (ann_return - rf_daily * 252) / ann_vol if ann_vol > 1e-10 else np.nan
    max_dd     = _max_drawdown(port_ret.values)
    calmar     = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else np.nan

    var95  = compute_all_var(returns_usd, valid, conf=0.95, horizon=1,  n_sim=n_sim)
    var99  = compute_all_var(returns_usd, valid, conf=0.99, horizon=1,  n_sim=n_sim)
    var10  = compute_all_var(returns_usd, valid, conf=0.95, horizon=10, n_sim=n_sim)

    beta_spy    = _ols_beta(port_ret, bench_spy)    if bench_spy    is not None else None
    beta_imoex  = _ols_beta(port_ret, bench_imoex) if bench_imoex  is not None else None

    info_ratio: Optional[float] = None
    if bench_spy is not None:
        diff = port_ret.align(bench_spy, join="inner")
        excess = diff[0] - diff[1]
        tr_err = float(excess.std() * np.sqrt(252))
        ann_exc = float(excess.mean() * 252)
        info_ratio = ann_exc / tr_err if tr_err > 1e-10 else np.nan

    return PortfolioMetrics(
        name=name,
        weights=valid,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        var_95_1d=var95.mc_t_var,
        es_95_1d=var95.mc_t_es,
        var_99_1d=var99.mc_t_var,
        var_95_10d=var10.mc_t_var,
        beta_spy=beta_spy,
        beta_imoex=beta_imoex,
        info_ratio=info_ratio,
    )


def benchmark_metrics(
    name: str,
    bench_ret: pd.Series,
    rf_daily: float = 0.0,
) -> pd.Series:
    r = bench_ret.dropna().values
    ann_return = float(r.mean() * 252)
    ann_vol    = float(r.std()  * np.sqrt(252))
    sharpe     = (ann_return - rf_daily * 252) / ann_vol if ann_vol > 1e-10 else np.nan
    max_dd     = _max_drawdown(r)
    calmar     = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else np.nan
    var_95     = float(-np.nanquantile(r, 0.05))

    def _f(x: float) -> str:
        return f"{x:.4f}" if np.isfinite(x) else "—"

    return pd.Series({
        "Портфель":    name,
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Vol":    f"{ann_vol:.2%}",
        "Sharpe":      _f(sharpe),
        "Calmar":      _f(calmar),
        "Max DD":      f"{max_dd:.2%}",
        "VaR 95% 1d":  f"{var_95:.3%}",
        "ES 95% 1d":   "—",
        "VaR 99% 1d":  "—",
        "VaR 95% 10d": "—",
        "β(SPY)":      "1.000" if name == "SPY" else "—",
        "β(IMOEX)":    "1.000" if name == "IMOEX" else "—",
    })


def compare_all(
    returns_usd: pd.DataFrame,
    spy_ret: Optional[pd.Series] = None,
    imoex_ret: Optional[pd.Series] = None,
    rf_daily: float = 0.0,
    n_sim: int = 1000,
    extra_portfolios: Optional[dict[str, dict[str, float]]] = None,
) -> pd.DataFrame:
    rows: list[pd.Series] = []

    all_variants = dict(PORTFOLIO_VARIANTS)
    if extra_portfolios:
        all_variants.update(extra_portfolios)

    for name, w in all_variants.items():
        try:
            m = compute_metrics(name, w, returns_usd, spy_ret, imoex_ret, rf_daily, n_sim=n_sim)
            rows.append(m.to_series())
        except Exception as exc:
            logger.warning("Portfolio %s failed: %s", name, exc)

    if spy_ret is not None:
        try:
            rows.append(benchmark_metrics("SPY", spy_ret, rf_daily))
        except Exception as exc:
            logger.warning("SPY benchmark failed: %s", exc)

    if imoex_ret is not None:
        try:
            rows.append(benchmark_metrics("IMOEX (RUB)", imoex_ret, rf_daily))
        except Exception as exc:
            logger.warning("IMOEX benchmark failed: %s", exc)

    return pd.DataFrame(rows).reset_index(drop=True)


def rolling_performance(
    weights: dict[str, float],
    returns_usd: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Скользящие Sharpe и VaR за window торговых дней."""
    valid = {t: w for t, w in weights.items() if t in returns_usd.columns}
    w_arr = np.array(list(valid.values()), dtype=float)
    w_arr /= w_arr.sum()
    port_ret = returns_usd[list(valid)].values @ w_arr

    out_dates, sharpes, vols, vars_ = [], [], [], []
    for i in range(window, len(port_ret)):
        sl = port_ret[i - window:i]
        ann_r  = float(sl.mean() * 252)
        ann_v  = float(sl.std()  * np.sqrt(252))
        sh     = ann_r / ann_v if ann_v > 1e-10 else np.nan
        sigma  = float(ewma_volatility(sl)[-1])
        var    = float(sigma * stats.norm.ppf(0.95))
        out_dates.append(returns_usd.index[i])
        sharpes.append(sh)
        vols.append(ann_v)
        vars_.append(var)

    return pd.DataFrame({"sharpe": sharpes, "ann_vol": vols, "var_ewma": vars_}, index=out_dates)
