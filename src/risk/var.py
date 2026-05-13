"""VaR/ES: historical, EWMA-parametric, MC normal, MC Student-t; горизонт 1d или 10d overlapping."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

LAMBDA_EWMA: float = 0.94
N_SIM:       int   = 10_000
CONF_DEFAULT: float = 0.95



def fx_adjusted_returns(ret_rub: pd.Series, usdrub_log_return: pd.Series) -> pd.Series:
    """Лог-доходность в USD: r_USD ≈ r_RUB − Δln(USDRUB)."""
    aligned_rub, aligned_s = ret_rub.align(usdrub_log_return, join="inner")
    return (aligned_rub - aligned_s).dropna()


def portfolio_returns_usd(
    asset_returns_usd: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Дневная доходность портфеля по весам (лог-доходности в USD)."""
    tickers = [t for t in weights if t in asset_returns_usd.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    rets = (
        asset_returns_usd[tickers]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
        .fillna(0.0)
    )
    return rets.values @ w



def ewma_volatility(returns: pd.Series | np.ndarray, lam: float = LAMBDA_EWMA) -> np.ndarray:
    """EWMA σ по RiskMetrics; длина совпадает с входом."""
    r = np.asarray(returns, dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    n = len(r)
    var = np.empty(n)
    var[0] = r[0] ** 2
    for i in range(1, n):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i - 1] ** 2
    return np.sqrt(var)


def ewma_cov_matrix(returns: pd.DataFrame, lam: float = LAMBDA_EWMA) -> np.ndarray:
    """EWMA ковариация на последней дате."""
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0).values
    n, k = clean.shape
    if n < 2:
        cov = np.eye(k) * 1e-6
    else:
        init_rows = max(2, min(n, 252))
        cov = np.cov(clean[:init_rows].T, ddof=1)
    cov = np.where(np.isfinite(cov), cov, np.eye(k) * 1e-6)
    for i in range(n):
        row = clean[i]
        if np.isfinite(row).all():
            cov = lam * cov + (1 - lam) * np.outer(row, row)
    return cov



def fit_student_t_nu(returns: pd.Series | np.ndarray) -> float:
    """MLE ν для одномерного t; ν ∈ [2.1, 30]."""
    r = np.asarray(returns, dtype=float)
    r = (r - r.mean()) / (r.std() + 1e-10)

    def neg_loglik(nu: float) -> float:
        if nu <= 2:
            return 1e10
        return -np.sum(stats.t.logpdf(r, df=nu))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize_scalar(neg_loglik, bounds=(2.1, 30), method="bounded")
    return float(res.x)



@dataclass
class VaRResult:
    """Результаты VaR/ES для одного горизонта."""
    confidence:    float
    horizon_days:  int

    hist_var:    float = np.nan
    hist_es:     float = np.nan

    param_var:   float = np.nan
    param_es:    float = np.nan

    mc_norm_var: float = np.nan
    mc_norm_es:  float = np.nan

    mc_t_var:    float = np.nan
    mc_t_es:     float = np.nan

    nu_t:        float = np.nan

    sqrt10_param_var: float = np.nan

    def to_series(self) -> pd.Series:
        return pd.Series({
            "Historical VaR":       self.hist_var,
            "Historical ES":        self.hist_es,
            "Parametric VaR":       self.param_var,
            "Parametric ES":        self.param_es,
            "MC Normal VaR":        self.mc_norm_var,
            "MC Normal ES":         self.mc_norm_es,
            "MC Student-t VaR":     self.mc_t_var,
            "MC Student-t ES":      self.mc_t_es,
            "Student-t nu":         self.nu_t,
            "Basel sqrt10 VaR":     self.sqrt10_param_var,
        })



def _quantile_var_es(losses: np.ndarray, conf: float) -> tuple[float, float]:
    """VaR и ES из массива потерь (потери = положительные числа). VaR = квантиль conf; ES = E[L | L >= VaR]."""
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    var = float(np.quantile(x, conf))
    tail = x[x >= var]
    es = float(tail.mean()) if tail.size else var
    return var, es


def _overlapping_windows(returns: np.ndarray, h: int) -> np.ndarray:
    """Скользящая сумма h дневных доходностей."""
    n = len(returns)
    windows = np.array([returns[i:i+h].sum() for i in range(n - h + 1)])
    return windows



def historical_var(
    port_returns: np.ndarray,
    conf:         float = CONF_DEFAULT,
    horizon:      int   = 1,
) -> tuple[float, float]:
    """Historical VaR и ES (потери выражены как положительные числа)."""
    if horizon > 1:
        pnl = _overlapping_windows(port_returns, horizon)
    else:
        pnl = port_returns.copy()
    losses = -pnl
    return _quantile_var_es(losses, conf)



def parametric_var(
    port_returns: np.ndarray,
    conf:         float = CONF_DEFAULT,
    horizon:      int   = 1,
    lam:          float = LAMBDA_EWMA,
) -> tuple[float, float]:
    """Нормальный VaR/ES: σ из EWMA (1d) или std overlapping P&L (h>1)."""
    if horizon > 1:
        pnl = _overlapping_windows(port_returns, horizon)
        sigma = float(np.std(pnl, ddof=1))
    else:
        sigmas = ewma_volatility(port_returns, lam)
        sigma  = float(sigmas[-1])

    z   = stats.norm.ppf(conf)
    var = sigma * z
    es  = sigma * stats.norm.pdf(z) / (1 - conf)
    return float(var), float(es)


def parametric_var_sqrt10(port_returns: np.ndarray, conf: float = CONF_DEFAULT) -> float:
    """VaR 10d через √10 от однодневного parametric VaR."""
    var1, _ = parametric_var(port_returns, conf, horizon=1)
    return float(var1 * np.sqrt(10))



def mc_normal_var(
    asset_returns:   pd.DataFrame,
    weights:         dict[str, float],
    conf:            float = CONF_DEFAULT,
    horizon:         int   = 1,
    n_sim:           int   = N_SIM,
    seed:            int   = 42,
    lam:             float = LAMBDA_EWMA,
) -> tuple[float, float]:
    """MC VaR/ES, многомерная нормаль, ковариация EWMA."""
    tickers = [t for t in weights if t in asset_returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    rets = asset_returns[tickers].replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0)
    cov  = ewma_cov_matrix(rets, lam=lam)

    rng = np.random.default_rng(seed)
    L   = np.linalg.cholesky(cov + np.eye(len(w)) * 1e-10)
    z   = rng.standard_normal((n_sim * horizon, len(w)))
    sim = (z @ L.T).reshape(n_sim, horizon, len(w)).sum(axis=1)
    port_pnl = sim @ w

    losses = -port_pnl
    return _quantile_var_es(losses, conf)


def mc_student_t_var(
    asset_returns:  pd.DataFrame,
    weights:        dict[str, float],
    conf:           float = CONF_DEFAULT,
    horizon:        int   = 1,
    n_sim:          int   = N_SIM,
    seed:           int   = 42,
    lam:            float = LAMBDA_EWMA,
) -> tuple[float, float, float]:
    """MC VaR/ES, многомерный t; ν — MLE по истории портфеля. Возврат: (var, es, nu)."""
    tickers = [t for t in weights if t in asset_returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    rets = asset_returns[tickers].replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0)
    port_hist = rets.values @ w

    nu  = fit_student_t_nu(port_hist)
    cov = ewma_cov_matrix(rets, lam=lam)

    rng = np.random.default_rng(seed)
    L   = np.linalg.cholesky(cov + np.eye(len(w)) * 1e-10)

    total = n_sim * horizon
    chi2  = rng.chisquare(df=nu, size=total)
    z     = rng.standard_normal((total, len(w)))
    scale = np.sqrt(nu / chi2)[:, np.newaxis]
    draws = (z * scale) @ L.T
    sim   = draws.reshape(n_sim, horizon, len(w)).sum(axis=1)
    port_pnl = sim @ w

    losses = -port_pnl
    var, es = _quantile_var_es(losses, conf)
    return float(var), float(es), float(nu)



def compute_all_var(
    asset_returns_usd: pd.DataFrame,
    weights:           dict[str, float],
    conf:              float = CONF_DEFAULT,
    horizon:           int   = 1,
    n_sim:             int   = N_SIM,
    lam:               float = LAMBDA_EWMA,
) -> VaRResult:
    """Сводка VaR/ES по четырём методам для заданного горизонта."""
    tickers = [t for t in weights if t in asset_returns_usd.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    w_dict = dict(zip(tickers, w.tolist()))

    rets = asset_returns_usd[tickers].replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0)
    port_returns = rets.values @ w

    result = VaRResult(confidence=conf, horizon_days=horizon)

    result.hist_var, result.hist_es = historical_var(port_returns, conf, horizon)

    result.param_var, result.param_es = parametric_var(port_returns, conf, horizon, lam=lam)

    if horizon == 10:
        result.sqrt10_param_var = parametric_var_sqrt10(port_returns, conf)

    result.mc_norm_var, result.mc_norm_es = mc_normal_var(rets, w_dict, conf, horizon, n_sim, lam=lam)

    result.mc_t_var, result.mc_t_es, result.nu_t = mc_student_t_var(
        rets, w_dict, conf, horizon, n_sim, lam=lam
    )

    return result


def compute_var_timeseries(
    asset_returns_usd: pd.DataFrame,
    weights:           dict[str, float],
    conf:              float = CONF_DEFAULT,
    method:            Literal["historical", "parametric", "mc_normal", "mc_t"] = "historical",
    window:            int   = 252,
    n_sim:             int   = N_SIM,
    lam:               float = LAMBDA_EWMA,
) -> pd.Series:
    """Expanding-window VaR по датам (для бэктеста)."""
    tickers = [t for t in weights if t in asset_returns_usd.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    w_dict = dict(zip(tickers, w.tolist()))

    rets = (
        asset_returns_usd[tickers]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
        .fillna(0.0)
    )
    port_returns = rets.values @ w
    dates = rets.index

    var_series: dict[pd.Timestamp, float] = {}
    for i in range(window, len(port_returns)):
        hist = port_returns[:i]
        if method == "historical":
            v, _ = historical_var(hist, conf, 1)
        elif method == "parametric":
            v, _ = parametric_var(hist, conf, 1, lam=lam)
        elif method == "mc_normal":
            v, _ = mc_normal_var(rets.iloc[:i], w_dict, conf, 1, n_sim=n_sim, lam=lam)
        else:
            v, _, _ = mc_student_t_var(rets.iloc[:i], w_dict, conf, 1, n_sim=n_sim, lam=lam)
        var_series[dates[i]] = v

    return pd.Series(var_series, name=f"var_{method}_{int(conf*100)}")
