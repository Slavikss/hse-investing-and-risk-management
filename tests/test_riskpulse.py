"""Unit tests for RiskPulse risk engine and backtest logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.risk.var import (
    _quantile_var_es,
    compute_all_var,
    fx_adjusted_returns,
    historical_var,
    parametric_var,
    portfolio_returns_usd,
)
from src.risk.attribution import compute_component_var
from src.backtest.kupiec import kupiec_pof


def test_fx_adjusted_returns_sign():
    """USD value of RUB asset: r_USD ≈ r_RUB − r_S (S = RUB per USD)."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    ret_rub = pd.Series([0.01, -0.02, 0.0, 0.01, -0.01], index=idx)
    r_s = pd.Series([0.05, 0.0, -0.01, 0.02, 0.0], index=idx)  # log(S_t/S_{t-1})
    out = fx_adjusted_returns(ret_rub, r_s)
    expected = ret_rub - r_s
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_portfolio_returns_weights_sum():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 100), "B": rng.normal(0, 0.01, 100)},
        index=idx,
    )
    w = {"A": 0.25, "B": 0.75}
    pr = portfolio_returns_usd(df, w)
    manual = df["A"].values * 0.25 + df["B"].values * 0.75
    np.testing.assert_allclose(pr, manual, rtol=1e-10)


def test_quantile_var_es_consistency():
    rng = np.random.default_rng(1)
    losses = np.sort(np.abs(rng.standard_normal(10_000)))
    conf = 0.95
    var, es = _quantile_var_es(losses, conf)
    var2 = float(np.quantile(losses, conf))
    assert abs(var - var2) < 1e-9
    tail = losses[losses >= var]
    assert abs(es - tail.mean()) < 1e-9
    assert es >= var


def test_parametric_var_normal_sanity():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, size=5000)
    conf = 0.95
    var, es = parametric_var(r, conf, horizon=1)
    sigma = float(np.std(r, ddof=1))
    z = stats.norm.ppf(conf)
    expected_var = sigma * z
    # EWMA σ на последнем шаге близка к выборочной, но не совпадает — допуск шире
    assert abs(var - expected_var) / max(expected_var, 1e-6) < 0.25
    assert es >= var * 0.99


def test_component_var_sum_matches_portfolio_normal():
    rng = np.random.default_rng(3)
    n, k = 2000, 3
    z = rng.multivariate_normal(np.zeros(k), np.eye(k) * (0.01 ** 2), size=n)
    cols = ["A", "B", "C"]
    rets = pd.DataFrame(z, columns=cols)
    w = {"A": 0.2, "B": 0.3, "C": 0.5}
    conf = 0.95
    attr = compute_component_var(rets, w, conf)
    z_a = stats.norm.ppf(conf)
    wv = attr.weights
    cov = attr.cov_matrix
    sigma_p = float(np.sqrt(wv @ cov @ wv))
    assert abs(attr.portfolio_var - sigma_p * z_a) < 1e-6
    assert abs(attr.component_var.sum() - attr.portfolio_var) < 1e-5


def test_kupiec_unconditional_coverage_iid():
    rng = np.random.default_rng(4)
    conf = 0.95
    alpha = 1 - conf
    n = 8000
    exceed = (rng.random(n) < alpha).astype(int)
    res = kupiec_pof(exceed, conf, "synthetic")
    assert res.n_obs == n
    assert res.n_exceed == int(exceed.sum())
    assert abs(res.actual_rate - alpha) < 0.012


def test_compute_all_var_finite():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2019-01-01", periods=400, freq="B")
    rets = pd.DataFrame(
        rng.multivariate_normal(np.zeros(3), np.eye(3) * 1e-4, size=400),
        index=idx,
        columns=["SBER", "NVDA", "MSFT"],
    )
    w = {"SBER": 0.3, "NVDA": 0.4, "MSFT": 0.3}
    out = compute_all_var(rets, w, conf=0.95, horizon=1, n_sim=2000)
    assert np.isfinite(out.hist_var) and out.hist_var > 0
    assert np.isfinite(out.param_var) and out.param_var > 0
    assert np.isfinite(out.mc_norm_var)
    assert np.isfinite(out.mc_t_var)
    assert np.isfinite(out.nu_t)


def test_historical_var_horizon10_length():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.01, size=500)
    v1, _ = historical_var(r, 0.95, 1)
    v10, _ = historical_var(r, 0.95, 10)
    assert np.isfinite(v10)
    assert v10 > 0
