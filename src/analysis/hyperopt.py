"""Оптимизация гиперпараметров: EWMA λ, веса портфеля, окно CAPM."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.risk.var import ewma_volatility, parametric_var, historical_var

logger = logging.getLogger(__name__)


# ─── EWMA lambda optimisation ────────────────────────────────────────────────

@dataclass
class LambdaOptResult:
    best_lambda:      float
    best_actual_rate: float
    target_rate:      float
    grid:             pd.DataFrame


def optimize_lambda(
    port_returns: np.ndarray,
    conf: float = 0.95,
    lambdas: list[float] | None = None,
    min_train: int = 60,
) -> LambdaOptResult:
    """
    Перебирает значения λ EWMA и выбирает то, при котором доля exceedances
    в expanding-window бэктесте наиболее близка к (1 − conf).
    """
    if lambdas is None:
        lambdas = np.round(np.arange(0.88, 0.995, 0.01), 3).tolist()

    target = 1.0 - conf
    rows = []
    r = np.asarray(port_returns, dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)

    for lam in lambdas:
        n_exc = 0
        n_obs = 0
        for i in range(min_train, len(r)):
            hist = r[:i]
            sigmas = ewma_volatility(hist, lam)
            sigma  = float(sigmas[-1])
            var    = sigma * stats.norm.ppf(conf)
            if -r[i] > var:
                n_exc += 1
            n_obs += 1
        actual = n_exc / n_obs if n_obs > 0 else np.nan
        rows.append({"lambda": lam, "actual_rate": actual, "target_rate": target, "error": abs(actual - target)})

    df = pd.DataFrame(rows)
    best_idx = df["error"].idxmin()
    return LambdaOptResult(
        best_lambda=float(df.loc[best_idx, "lambda"]),
        best_actual_rate=float(df.loc[best_idx, "actual_rate"]),
        target_rate=target,
        grid=df,
    )


# ─── CAPM window optimisation ────────────────────────────────────────────────

@dataclass
class WindowOptResult:
    best_window:  int
    best_avg_r2:  float
    grid:         pd.DataFrame


def optimize_capm_window(
    asset_ret:  pd.Series,
    market_ret: pd.Series,
    windows:    list[int] | None = None,
) -> WindowOptResult:
    """
    Ищет окно rolling OLS, при котором средний R² наибольший.
    Использует только алгоритм lstsq без внешних зависимостей.
    """
    if windows is None:
        windows = [60, 90, 120, 180, 252]

    y, x = asset_ret.align(market_ret, join="inner")
    y, x = y.dropna(), x.dropna()
    common = y.index.intersection(x.index)
    y, x = y.loc[common].values, x.loc[common].values

    rows = []
    for w in windows:
        if len(y) < w + 20:
            rows.append({"window": w, "avg_r2": np.nan, "std_beta": np.nan})
            continue
        r2s, betas = [], []
        for i in range(w, len(y)):
            yw = y[i - w:i]
            xw = x[i - w:i]
            X  = np.column_stack([np.ones(w), xw])
            coef, *_ = np.linalg.lstsq(X, yw, rcond=None)
            pred   = X @ coef
            ss_res = float(np.sum((yw - pred) ** 2))
            ss_tot = float(np.sum((yw - yw.mean()) ** 2))
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
            r2s.append(r2)
            betas.append(float(coef[1]))
        rows.append({"window": w, "avg_r2": float(np.mean(r2s)), "std_beta": float(np.std(betas))})

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["avg_r2"])
    if valid.empty:
        return WindowOptResult(best_window=windows[-1], best_avg_r2=float("nan"), grid=df)
    best_idx = valid["avg_r2"].idxmax()
    return WindowOptResult(
        best_window=int(df.loc[best_idx, "window"]),
        best_avg_r2=float(df.loc[best_idx, "avg_r2"]),
        grid=df,
    )


# ─── Portfolio weight optimisation ───────────────────────────────────────────

@dataclass
class WeightOptResult:
    objective:      str
    optimal_weights: dict[str, float]
    metric_value:   float
    converged:      bool


def optimize_weights(
    returns_usd: pd.DataFrame,
    objective:   str = "sharpe",
    conf:        float = 0.95,
    min_w:       float = 0.02,
    max_w:       float = 0.60,
) -> WeightOptResult:
    """
    Оптимизация весов портфеля.

    Цели (objective):
      'sharpe'    — максимизация Sharpe Ratio
      'min_var'   — минимизация 1d параметрического VaR
      'min_vol'   — минимизация годовой волатильности
      'min_es'    — минимизация Expected Shortfall (исторический)
      'calmar'    — максимизация Calmar Ratio (Ann.Return / MaxDD)
    """
    tickers = [c for c in returns_usd.columns]
    n       = len(tickers)
    rets    = returns_usd[tickers].replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0).values

    def _port(w: np.ndarray) -> np.ndarray:
        return rets @ w

    def neg_sharpe(w: np.ndarray) -> float:
        p   = _port(w)
        ann = p.mean() * 252
        vol = p.std() * np.sqrt(252)
        return -ann / vol if vol > 1e-10 else 0.0

    def par_var(w: np.ndarray) -> float:
        p = _port(w)
        v, _ = parametric_var(p, conf, horizon=1)
        return float(v)

    def ann_vol(w: np.ndarray) -> float:
        return float(_port(w).std() * np.sqrt(252))

    def hist_es(w: np.ndarray) -> float:
        p      = _port(w)
        losses = -p
        var    = float(np.quantile(losses, conf))
        tail   = losses[losses >= var]
        return float(tail.mean()) if tail.size else float(var)

    def neg_calmar(w: np.ndarray) -> float:
        p     = _port(w)
        ann   = p.mean() * 252
        cum   = np.exp(np.cumsum(p))
        peak  = np.maximum.accumulate(cum)
        dd    = (cum - peak) / np.where(peak > 0, peak, 1.0)
        maxdd = abs(dd.min())
        return -ann / maxdd if maxdd > 1e-10 else 0.0

    objectives = {
        "sharpe":  neg_sharpe,
        "min_var": par_var,
        "min_vol": ann_vol,
        "min_es":  hist_es,
        "calmar":  neg_calmar,
    }
    if objective not in objectives:
        raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(objectives)}")

    fn          = objectives[objective]
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(min_w, max_w)] * n
    w0          = np.ones(n) / n

    result = optimize.minimize(
        fn, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    w_opt = result.x / result.x.sum()
    metric = -result.fun if objective in ("sharpe", "calmar") else result.fun

    return WeightOptResult(
        objective=objective,
        optimal_weights=dict(zip(tickers, w_opt.round(6).tolist())),
        metric_value=float(metric),
        converged=bool(result.success),
    )


def run_all_optimisations(
    returns_usd: pd.DataFrame,
    port_returns: np.ndarray,
    conf: float = 0.95,
    spy_ret: pd.Series | None = None,
    main_asset: str = "SBER",
) -> dict:
    """Запускает все оптимизации и возвращает словарь результатов."""
    out: dict = {}

    logger.info("Optimising EWMA lambda...")
    out["lambda"] = optimize_lambda(port_returns, conf)

    for obj in ("sharpe", "min_var", "min_vol", "min_es", "calmar"):
        logger.info("Optimising weights: %s", obj)
        try:
            out[f"weights_{obj}"] = optimize_weights(returns_usd, objective=obj, conf=conf)
        except Exception as exc:
            logger.warning("Weight opt %s failed: %s", obj, exc)

    if spy_ret is not None and main_asset in returns_usd.columns:
        logger.info("Optimising CAPM window for %s vs SPY...", main_asset)
        out["capm_window"] = optimize_capm_window(returns_usd[main_asset], spy_ret)

    return out
