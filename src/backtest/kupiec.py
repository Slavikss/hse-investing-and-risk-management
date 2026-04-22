"""Kupiec (LR_uc), Christoffersen (LR_ind), совместный LR_cc по бинарным exceedances."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class KupiecResult:
    method:       str
    confidence:   float
    n_obs:        int
    n_exceed:     int
    expected_rate: float
    actual_rate:  float

    lr_uc:        float
    p_uc:         float
    reject_uc:    bool

    lr_ind:       float
    p_ind:        float
    reject_ind:   bool

    lr_cc:        float
    p_cc:         float
    reject_cc:    bool

    pi:           float = 0.0
    pi_01:        float = 0.0
    pi_11:        float = 0.0
    n00: int = 0; n01: int = 0; n10: int = 0; n11: int = 0

    def summary(self) -> str:
        lines = [
            f"Method:    {self.method}",
            f"T={self.n_obs}, exceedances={self.n_exceed} ({self.actual_rate:.2%} vs expected {self.expected_rate:.2%})",
            f"LR_uc={self.lr_uc:.4f}  p={self.p_uc:.4f}  {'REJECT' if self.reject_uc else 'pass'}",
            f"LR_ind={self.lr_ind:.4f} p={self.p_ind:.4f}  {'REJECT' if self.reject_ind else 'pass'}",
            f"LR_cc={self.lr_cc:.4f}  p={self.p_cc:.4f}  {'REJECT' if self.reject_cc else 'pass'}",
        ]
        return "\n".join(lines)


def _safe_log(x: float) -> float:
    return float(np.log(x)) if x > 1e-300 else -700.0


def kupiec_pof(
    exceedances: np.ndarray,
    confidence:  float,
    method_name: str = "VaR",
) -> KupiecResult:
    """POF (LR_uc) и independence (LR_ind) по ряду exceedances 0/1."""
    exc = np.asarray(exceedances, dtype=int)
    T   = len(exc)
    n   = int(exc.sum())
    alpha  = 1.0 - confidence
    p_hat  = n / T if T > 0 else 0.0

    if n == 0:
        lr_uc = -2.0 * (T * _safe_log(1 - alpha))
    elif n == T:
        lr_uc = -2.0 * (T * _safe_log(alpha))
    else:
        ll_null = (T - n) * _safe_log(1 - alpha) + n * _safe_log(alpha)
        ll_alt  = (T - n) * _safe_log(1 - p_hat) + n * _safe_log(p_hat)
        lr_uc   = -2.0 * (ll_null - ll_alt)

    p_uc     = float(stats.chi2.sf(lr_uc, df=1))
    reject_uc = p_uc < 0.05

    n00 = int(((exc[:-1] == 0) & (exc[1:] == 0)).sum())
    n01 = int(((exc[:-1] == 0) & (exc[1:] == 1)).sum())
    n10 = int(((exc[:-1] == 1) & (exc[1:] == 0)).sum())
    n11 = int(((exc[:-1] == 1) & (exc[1:] == 1)).sum())

    n_trans = n00 + n01 + n10 + n11
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / n_trans if n_trans > 0 else 0.0

    def _ll_ind(p0: float, p1: float) -> float:
        ll = 0.0
        if n00 > 0: ll += n00 * _safe_log(1 - p0)
        if n01 > 0: ll += n01 * _safe_log(p0)
        if n10 > 0: ll += n10 * _safe_log(1 - p1)
        if n11 > 0: ll += n11 * _safe_log(p1)
        return ll

    degenerate = n_trans == 0 or (n00 + n01) == 0 or (n10 + n11) == 0
    if degenerate:
        lr_ind, p_ind, reject_ind = 0.0, 1.0, False
    else:
        ll_unrest = _ll_ind(pi_01, pi_11)
        ll_restr = _ll_ind(pi, pi)
        lr_ind = max(0.0, -2.0 * (ll_restr - ll_unrest))
        p_ind = float(stats.chi2.sf(lr_ind, df=1))
        reject_ind = p_ind < 0.05

    lr_cc    = lr_uc + lr_ind
    p_cc     = float(stats.chi2.sf(lr_cc, df=2))
    reject_cc = p_cc < 0.05

    return KupiecResult(
        method=method_name,
        confidence=confidence,
        n_obs=T,
        n_exceed=n,
        expected_rate=alpha,
        actual_rate=p_hat,
        lr_uc=lr_uc, p_uc=p_uc, reject_uc=reject_uc,
        lr_ind=lr_ind, p_ind=p_ind, reject_ind=reject_ind,
        lr_cc=lr_cc, p_cc=p_cc, reject_cc=reject_cc,
        pi=pi, pi_01=pi_01, pi_11=pi_11,
        n00=n00, n01=n01, n10=n10, n11=n11,
    )


def run_all_kupiec_tests(
    realized_pnl:    np.ndarray,
    var_series_dict: dict[str, np.ndarray],
    confidence:      float = 0.95,
) -> list[KupiecResult]:
    """KupiecResult по каждому методу: exceedance = (−P&L) > VaR."""
    results = []
    for method, var_array in var_series_dict.items():
        min_len = min(len(realized_pnl), len(var_array))
        pnl = realized_pnl[-min_len:]
        var = var_array[-min_len:]

        exceed = ((-pnl) > var).astype(int)
        res = kupiec_pof(exceed, confidence, method_name=method)
        results.append(res)
    return results


def kupiec_results_to_df(results: list[KupiecResult]) -> "pd.DataFrame":
    """Сводная таблица результатов всех тестов."""
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "Метод":          r.method,
            "T":              r.n_obs,
            "Exceedances":    r.n_exceed,
            "Ожид. %":        round(r.expected_rate * 100, 1),
            "Факт. %":        round(r.actual_rate * 100, 1),
            "LR_uc":          round(r.lr_uc, 3),
            "p_uc":           round(r.p_uc, 4),
            "Reject_uc":      r.reject_uc,
            "LR_ind":         round(r.lr_ind, 3),
            "p_ind":          round(r.p_ind, 4),
            "Reject_ind":     r.reject_ind,
            "LR_cc":          round(r.lr_cc, 3),
            "p_cc":           round(r.p_cc, 4),
            "Reject_cc":      r.reject_cc,
        })
    return pd.DataFrame(rows)
