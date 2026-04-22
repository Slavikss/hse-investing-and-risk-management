"""Компонентный VaR, FX-декомпозиция RU, подбор снижения позиции под лимит VaR."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RU_TICKERS     = ["SBER", "LKOH", "GAZP"]
GLOBAL_TICKERS = ["NVDA", "MSFT", "AAPL"]



@dataclass
class AttributionResult:
    tickers:        list[str]
    weights:        np.ndarray
    portfolio_var:  float
    component_var:  np.ndarray
    component_pct:  np.ndarray
    cov_matrix:     np.ndarray
    confidence:     float = 0.95
    fx_component:     np.ndarray = field(default_factory=lambda: np.array([]))
    equity_component: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "ticker":        self.tickers,
            "weight":        self.weights,
            "component_var": self.component_var,
            "component_pct": self.component_pct * 100,
        })
        if len(self.fx_component) == len(self.tickers):
            df["fx_component"]     = self.fx_component
            df["equity_component"] = self.equity_component
        return df


def compute_component_var(
    asset_returns_usd: pd.DataFrame,
    weights:           dict[str, float],
    conf:              float = 0.95,
) -> AttributionResult:
    """Компонентный VaR из выборочной ковариации и нормального квантиля."""
    from scipy import stats

    tickers = [t for t in weights if t in asset_returns_usd.columns]
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()

    rets = asset_returns_usd[tickers].replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0).values
    cov  = np.cov(rets.T, ddof=1)

    sigma_p = float(np.sqrt(w @ cov @ w))
    z_alpha = stats.norm.ppf(conf)

    marginal = cov @ w / sigma_p
    component = w * marginal
    component_pct = component / component.sum()

    component_var = component * z_alpha

    return AttributionResult(
        tickers=tickers,
        weights=w,
        confidence=conf,
        portfolio_var=sigma_p * z_alpha,
        component_var=component_var,
        component_pct=component_pct,
        cov_matrix=cov,
    )



def add_fx_decomposition(
    attribution:       AttributionResult,
    asset_returns_usd: pd.DataFrame,
    usdrub_returns:    pd.Series,
) -> AttributionResult:
    """Для RU-тикеров: доля компонентного VaR от корреляции с USDRUB и остаток «equity»."""
    from scipy import stats

    tickers = attribution.tickers
    w       = attribution.weights
    cov     = attribution.cov_matrix
    conf    = float(attribution.confidence)
    z_alpha = stats.norm.ppf(conf)
    sigma_p = attribution.portfolio_var / z_alpha

    rets   = asset_returns_usd[tickers].replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0)
    fx_ret = usdrub_returns.replace([np.inf, -np.inf], np.nan).fillna(0).rename("fx")

    aligned = rets.join(fx_ret, how="inner").dropna()
    fx_vals = aligned["fx"].values

    fx_component     = np.zeros(len(tickers))
    equity_component = np.zeros(len(tickers))

    for i, ticker in enumerate(tickers):
        if ticker in RU_TICKERS:
            asset_vals = aligned[ticker].values
            cov_with_fx = float(np.cov(asset_vals, fx_vals, ddof=1)[0, 1])
            rc_fx = w[i] * cov_with_fx / sigma_p * z_alpha
            fx_component[i]     = rc_fx
            equity_component[i] = attribution.component_var[i] - rc_fx
        else:
            fx_component[i]     = 0.0
            equity_component[i] = attribution.component_var[i]

    attribution.fx_component     = fx_component
    attribution.equity_component = equity_component
    return attribution



def check_var_limit(
    portfolio_var:  float,
    limit:          float,
    attribution:    AttributionResult,
) -> dict:
    """Флаг превышения лимита, лидер по вкладу, грубая оценка снижения веса."""
    breach = portfolio_var > limit
    result: dict = {
        "breach":        breach,
        "portfolio_var": portfolio_var,
        "limit":         limit,
        "excess":        max(0.0, portfolio_var - limit),
    }

    if not breach:
        result["top_contributor"] = None
        result["suggested_weight_reduction"] = None
        return result

    top_idx = int(np.argmax(np.abs(attribution.component_var)))
    top_ticker = attribution.tickers[top_idx]
    result["top_contributor"] = top_ticker

    rc_top = float(attribution.component_var[top_idx])
    w_top  = float(attribution.weights[top_idx])
    if rc_top > 0 and w_top > 0:
        sens = rc_top / w_top
        delta_w = result["excess"] / sens if sens != 0 else 0.0
        new_w = max(0.0, w_top - delta_w)
        result["suggested_weight_reduction"] = {
            "ticker":       top_ticker,
            "current_w":    round(w_top, 4),
            "suggested_w":  round(new_w, 4),
            "delta_w":      round(delta_w, 4),
        }
    else:
        result["suggested_weight_reduction"] = None

    return result



def full_attribution(
    asset_returns_usd: pd.DataFrame,
    weights:           dict[str, float],
    usdrub_returns:    pd.Series,
    conf:              float = 0.95,
    var_limit:         float = 0.02,
) -> tuple[AttributionResult, dict]:
    """Компонентный VaR, FX-декомпозиция, проверка лимита."""
    attr = compute_component_var(asset_returns_usd, weights, conf)
    attr = add_fx_decomposition(attr, asset_returns_usd, usdrub_returns)
    limit_check = check_var_limit(attr.portfolio_var, var_limit, attr)
    return attr, limit_check
