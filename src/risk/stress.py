"""Исторические сценарии, гипотетические шоки, ковариации normal/crisis."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "COVID":      ("2020-02-17", "2020-03-23"),
    "RateHike":   ("2022-03-16", "2022-12-31"),
    "Sanctions":  ("2022-02-21", "2022-02-25"),
}

NORMAL_EXCLUDE = [w for w in CRISIS_WINDOWS.values()]



def compute_covariance_matrices(
    asset_returns_usd: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Вычисляет ковариационные матрицы на двух режимах:
      'normal' : полная история минус кризисные окна
      'crisis' : только COVID Feb-Apr 2020 + Sanctions Feb-Mar 2022

    Returns dict с ключами 'normal', 'crisis', 'full'.
    """
    rets = asset_returns_usd.dropna(how="all").fillna(0)

    # Crisis mask
    crisis_mask = pd.Series(False, index=rets.index)
    for start, end in CRISIS_WINDOWS.values():
        crisis_mask |= (rets.index >= start) & (rets.index <= end)

    normal_rets = rets[~crisis_mask]
    crisis_rets  = rets[crisis_mask]

    cov_full   = np.cov(rets.values.T, ddof=1)
    k = rets.shape[1]
    if len(normal_rets) > k + 1:
        cov_normal = np.cov(normal_rets.values.T, ddof=1)
    else:
        cov_normal = cov_full
    if len(crisis_rets) > k + 1:
        cov_crisis = np.cov(crisis_rets.values.T, ddof=1)
    else:
        cov_crisis = cov_full

    return {
        "full":   cov_full,
        "normal": cov_normal,
        "crisis": cov_crisis,
    }


def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    """Нормализует ковариацию в матрицу корреляций (устойчиво к нулевой дисперсии)."""
    d = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    out = cov / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return np.clip(out, -1.0, 1.0)


def avg_correlation(cov: np.ndarray) -> float:
    """Средняя off-diagonal корреляция (игнорируем nan/inf)."""
    corr = corr_from_cov(cov)
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    vals = corr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanmean(vals))



@dataclass
class ScenarioResult:
    name:         str
    period:       str
    pnl_pct:      float        # % от стоимости портфеля
    pnl_abs:      float        # в USD (при portfolio_value=1)
    contrib:      dict[str, float] = field(default_factory=dict)  # вклад каждого актива
    available_tickers: list[str]   = field(default_factory=list)


def apply_historical_scenario(
    scenario_name:     str,
    asset_returns_usd: pd.DataFrame,
    weights:           dict[str, float],
) -> ScenarioResult:
    """
    Применяет исторический кризисный сценарий к текущему портфелю.

    Логика:
      1. Берём реальные доходности за кризисный период
      2. Нормируем на длину периода → кумулятивная P&L за весь эпизод
      3. Взвешиваем по портфельным весам
    """
    if scenario_name not in CRISIS_WINDOWS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(CRISIS_WINDOWS)}")

    start, end = CRISIS_WINDOWS[scenario_name]

    tickers = [t for t in weights if t in asset_returns_usd.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    period_rets = asset_returns_usd.loc[start:end, tickers].dropna(how="all")

    if period_rets.empty:
        logger.warning("Scenario %s: no data for period %s–%s", scenario_name, start, end)
        return ScenarioResult(
            name=scenario_name, period=f"{start}–{end}",
            pnl_pct=np.nan, pnl_abs=np.nan,
        )

    cum_rets = period_rets.fillna(0).sum(axis=0)  # сумма лог-доходностей
    available = list(period_rets.columns)

    pnl_per_asset = dict(zip(tickers, (cum_rets * w).tolist()))
    pnl_total = float(sum(pnl_per_asset.values()))

    return ScenarioResult(
        name=scenario_name,
        period=f"{start}–{end}",
        pnl_pct=pnl_total * 100,
        pnl_abs=pnl_total,
        contrib=pnl_per_asset,
        available_tickers=available,
    )


def run_all_historical_scenarios(
    asset_returns_usd: pd.DataFrame,
    weights:           dict[str, float],
) -> list[ScenarioResult]:
    """Прогоняет все три исторических сценария."""
    return [
        apply_historical_scenario(name, asset_returns_usd, weights)
        for name in CRISIS_WINDOWS
    ]



@dataclass
class HypotheticalShock:
    name:        str
    description: str
    factor:      str
    delta:       float       # изменение фактора (в %)
    sensitivities: dict[str, float]  # {ticker: dP/dFactor} — линейное приближение


def build_hypothetical_shocks(
    asset_returns_usd: pd.DataFrame,
    usdrub_returns:    pd.Series,
    brent_returns:     pd.Series | None = None,
) -> list[HypotheticalShock]:
    """
    Строит три гипотетических шока.

    Sensitivities:
      FX shock    — регрессия dollar-returns RU активов на FX доходность
      Brent shock — регрессия LKOH/GAZP на Brent returns (если доступно)
      Rate shock  — приближение через duration-proxy для growth акций
    """
    def _ols_slope(yv: np.ndarray, xv: np.ndarray) -> float:
        """Регрессия y ~ const + x, возвращает наклон по x."""
        X = np.column_stack([np.ones(len(xv)), xv.astype(float)])
        yv = yv.astype(float)
        m = np.isfinite(X).all(axis=1) & np.isfinite(yv)
        if int(m.sum()) < 5:
            return 0.0
        coef, *_ = np.linalg.lstsq(X[m], yv[m], rcond=None)
        return float(coef[1])

    shocks = []
    tickers = list(asset_returns_usd.columns)

    # Положительный r_FX = рубль ослабевает (1 USD стоит дороже)
    # RU экспортёры (LKOH, GAZP) выигрывают, SBER нейтрален
    fx_sens = {}
    aligned = asset_returns_usd.join(usdrub_returns.rename("fx"), how="inner").dropna()
    for ticker in tickers:
        if ticker in aligned.columns and ticker != "fx":
            try:
                coef = _ols_slope(aligned[ticker].values, aligned["fx"].values)
            except Exception:
                coef = 0.0
            fx_sens[ticker] = float(coef)

    shocks.append(HypotheticalShock(
        name="FX_shock_+20pct",
        description="USD/RUB +20% (рубль слабеет)",
        factor="USDRUB",
        delta=0.20,
        sensitivities=fx_sens,
    ))

    brent_sens: dict[str, float] = {}
    if brent_returns is not None:
        aligned_b = asset_returns_usd.join(brent_returns.rename("brent"), how="inner").dropna()
        for ticker in ["LKOH", "GAZP", "SBER"]:
            if ticker in aligned_b.columns:
                try:
                    coef = _ols_slope(aligned_b[ticker].values, aligned_b["brent"].values)
                except Exception:
                    coef = 0.0
                brent_sens[ticker] = float(coef)
        # Global акции имеют малую прямую экспозицию к нефти
        for t in ["NVDA", "MSFT", "AAPL"]:
            brent_sens[t] = 0.0

    shocks.append(HypotheticalShock(
        name="Brent_-30pct",
        description="Нефть Brent −30%",
        factor="Brent",
        delta=-0.30,
        sensitivities=brent_sens,
    ))

    # Приближение: duration-proxy для growth акций (P/E > 30)
    # NVDA, MSFT — высокий мультипликатор → ставки давят; AAPL умеренно
    rate_sens = {
        "NVDA": -2.5,   # высокий P/E → чувствительность как у 2.5-летней облигации
        "MSFT": -1.8,
        "AAPL": -1.2,
        "SBER": -0.5,   # банк: ставки двоякие; net interest margin vs credit risk
        "LKOH": -0.3,
        "GAZP": -0.3,
    }
    shocks.append(HypotheticalShock(
        name="FedRate_+200bp",
        description="Ставка ФРС +200bp (rate hike proxy)",
        factor="FEDFUNDS",
        delta=0.02,   # 200bp = 0.02
        sensitivities=rate_sens,
    ))

    return shocks


def apply_hypothetical_shock(
    shock:   HypotheticalShock,
    weights: dict[str, float],
) -> dict[str, float]:
    """
    Линейное P&L: ΔP = Σ wᵢ × sensitivity_i × Δfactor

    Returns: {ticker: pnl_contribution, "total": total_pnl}
    """
    tickers = [t for t in weights if t in shock.sensitivities]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    result = {}
    total = 0.0
    for ticker, wi in zip(tickers, w):
        pnl = wi * shock.sensitivities[ticker] * shock.delta
        result[ticker] = float(pnl)
        total += pnl
    result["total"] = float(total)
    return result



def stress_summary_table(
    historical_results:   list[ScenarioResult],
    hypothetical_results: list[dict],
    shock_names:          list[str],
) -> pd.DataFrame:
    """Строит сводную таблицу P&L по всем сценариям."""
    rows = []
    for res in historical_results:
        rows.append({
            "Сценарий":    res.name,
            "Тип":         "Исторический",
            "Период":      res.period,
            "P&L (%)":     round(res.pnl_pct, 2),
        })
    for name, res_dict in zip(shock_names, hypothetical_results):
        rows.append({
            "Сценарий":    name,
            "Тип":         "Гипотетический",
            "Период":      "—",
            "P&L (%)":     round(res_dict.get("total", np.nan) * 100, 2),
        })
    return pd.DataFrame(rows)
