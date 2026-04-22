"""Расширенные тесты: CAPM, stress, DB, ingestion, runner, CLI, VaR, attribution, Kupiec."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# --- CAPM -----------------------------------------------------------------
def _business_idx(n: int, start: str = "2018-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def test_capm_log_returns_and_rf_daily():
    from src.risk.capm import _log_returns, _annual_rate_to_daily, _add_constant_series

    idx = _business_idx(20)
    p = pd.DataFrame({"SBER": np.linspace(100, 110, 20)}, index=idx)
    lr = _log_returns(p)
    assert lr.shape[0] == 19
    rate = pd.Series([10.0, 11.0], index=idx[:2])
    rd = _annual_rate_to_daily(rate.reindex(idx).ffill())
    assert np.isfinite(rd.iloc[-1])
    s = pd.Series(np.random.randn(10), index=_business_idx(10))
    X = _add_constant_series(s, "mkt")
    assert "const" in X.columns and "mkt" in X.columns


def test_rolling_ols_single_short_returns_empty():
    from src.risk.capm import _rolling_ols_single

    idx = _business_idx(50)
    y = pd.Series(np.random.randn(50) * 0.01, index=idx)
    X = pd.DataFrame({"const": 1.0, "x": np.random.randn(50) * 0.01}, index=idx)
    out = _rolling_ols_single(y, X, window=252, min_obs=60)
    assert out.empty


def test_rolling_ols_single_nonempty():
    from src.risk.capm import _rolling_ols_single

    idx = _business_idx(300)
    x = np.random.default_rng(1).normal(0, 0.01, len(idx))
    y = 0.5 * x + np.random.default_rng(2).normal(0, 0.005, len(idx))
    X = pd.DataFrame({"const": 1.0, "x": x}, index=idx)
    ys = pd.Series(y, index=idx)
    out = _rolling_ols_single(ys, X, window=120, min_obs=40)
    assert not out.empty
    assert set(out.columns) == {"alpha", "beta", "r_squared"}


def test_rolling_ff3_and_capm_pipelines():
    from src.risk import capm

    capm._french_cache = None
    idx = _business_idx(400)
    rng = np.random.default_rng(3)
    rub = pd.DataFrame(
        {
            "SBER": 250 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx)))),
            "LKOH": 4000 * np.exp(np.cumsum(rng.normal(0.0002, 0.014, len(idx)))),
            "GAZP": 160 * np.exp(np.cumsum(rng.normal(0.0001, 0.016, len(idx)))),
        },
        index=idx,
    )
    imoex = pd.Series(2800 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(idx)))), index=idx)
    cbr = pd.Series(7.5, index=idx)
    ru = capm.rolling_capm_ru(rub, imoex, cbr, window=120)
    assert not ru.empty
    assert ru["cluster"].eq("RU").all()

    usd = pd.DataFrame(
        {
            "NVDA": 500 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(idx)))),
            "MSFT": 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx)))),
            "AAPL": 180 * np.exp(np.cumsum(rng.normal(0.0002, 0.014, len(idx)))),
        },
        index=idx,
    )
    spy = pd.Series(400 * np.exp(np.cumsum(rng.normal(0.00025, 0.013, len(idx)))), index=idx)
    fed = pd.Series(2.5, index=idx)
    gl = capm.rolling_capm_global(usd, spy, fed, window=120)
    assert not gl.empty

    cache = Path(__file__).resolve().parents[1] / "data" / "ff3_daily.parquet"
    if not cache.exists():
        pytest.skip("data/ff3_daily.parquet отсутствует")
    ff = capm.load_french_factors(cache)
    ff3 = ff.reindex(idx).dropna()
    usd2 = usd.reindex(ff3.index).dropna(how="all")
    ff3 = ff3.reindex(usd2.index).dropna()
    usd2 = usd2.loc[ff3.index]
    if len(usd2) < 200:
        pytest.skip("мало пересечений с FF3")
    out_ff = capm.rolling_ff3_global(usd2, ff3, window=120)
    assert out_ff.empty or "beta_smb" in out_ff.columns

    latest = capm.latest_betas(ru, gl)
    assert not latest.empty


def test_latest_betas_empty_inputs():
    from src.risk.capm import latest_betas

    assert latest_betas(pd.DataFrame(), pd.DataFrame()).empty


def test_load_french_factors_uses_parquet(tmp_path):
    from src.risk import capm

    capm._french_cache = None
    src = Path(__file__).resolve().parents[1] / "data" / "ff3_daily.parquet"
    if not src.exists():
        pytest.skip("нет ff3_daily.parquet")
    df = pd.read_parquet(src).head(100)
    p = tmp_path / "ff3.parquet"
    df.to_parquet(p)
    capm._french_cache = None
    got = capm.load_french_factors(p)
    assert len(got) >= 50


# --- Stress ----------------------------------------------------------------
def test_stress_covariance_corr_avg():
    from src.risk.stress import (
        apply_historical_scenario,
        apply_hypothetical_shock,
        avg_correlation,
        build_hypothetical_shocks,
        compute_covariance_matrices,
        corr_from_cov,
        run_all_historical_scenarios,
        stress_summary_table,
    )

    idx = pd.bdate_range("2019-01-01", periods=800)
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(
        rng.normal(0, 0.01, (len(idx), 3)),
        index=idx,
        columns=["SBER", "NVDA", "MSFT"],
    ).astype(np.float64)
    mats = compute_covariance_matrices(rets)
    assert set(mats) == {"full", "normal", "crisis"}
    c = corr_from_cov(mats["full"])
    assert c.shape[0] == c.shape[1]
    ac = avg_correlation(mats["full"])
    assert np.isfinite(ac)

    w = {"SBER": 0.5, "NVDA": 0.25, "MSFT": 0.25}
    scenarios = run_all_historical_scenarios(rets, w)
    assert len(scenarios) == 3

    with pytest.raises(ValueError):
        apply_historical_scenario("UNKNOWN", rets, w)

    fx = pd.Series(rng.normal(0, 0.005, len(idx)), index=idx)
    brent = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    shocks = build_hypothetical_shocks(rets, fx, brent)
    assert len(shocks) >= 3
    res0 = apply_hypothetical_shock(shocks[0], w)
    assert "total" in res0

    hist = [apply_historical_scenario(n, rets, w) for n in ("COVID", "RateHike", "Sanctions")]
    hyp = [apply_hypothetical_shock(s, w) for s in shocks]
    tbl = stress_summary_table(hist, hyp, [s.name for s in shocks])
    assert not tbl.empty

    far = pd.bdate_range("2090-01-01", periods=30)
    empty_region = pd.DataFrame(rng.normal(0, 0.01, (len(far), 3)), index=far, columns=["SBER", "NVDA", "MSFT"])
    empty_sc = apply_historical_scenario("COVID", empty_region, w)
    assert np.isnan(empty_sc.pnl_pct)


def test_corr_from_cov_zero_variance():
    from src.risk.stress import corr_from_cov

    cov = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    c = corr_from_cov(cov)
    assert c.shape == (2, 2)


def test_avg_correlation_trivial():
    from src.risk.stress import avg_correlation

    assert np.isnan(avg_correlation(np.eye(1)))


# --- VaR -------------------------------------------------------------------
def test_var_helpers_and_mc():
    from src.risk.var import (
        VaRResult,
        _quantile_var_es,
        compute_all_var,
        compute_var_timeseries,
        ewma_cov_matrix,
        ewma_volatility,
        fit_student_t_nu,
        mc_normal_var,
        mc_student_t_var,
        parametric_var_sqrt10,
    )

    v, e = _quantile_var_es(np.array([]), 0.95)
    assert np.isnan(v) and np.isnan(e)

    r = np.random.default_rng(8).normal(0, 0.01, 400)
    sig = ewma_volatility(r)
    assert len(sig) == len(r)

    cov1 = ewma_cov_matrix(pd.DataFrame([[1.0]], columns=["a"]))
    assert cov1.shape == (1, 1)

    nu = fit_student_t_nu(r)
    assert 2.1 <= nu <= 30

    idx = pd.bdate_range("2019-01-01", periods=300)
    df = pd.DataFrame(
        np.random.default_rng(9).normal(0, 0.01, (300, 2)),
        index=idx,
        columns=["SBER", "NVDA"],
    )
    w = {"SBER": 0.6, "NVDA": 0.4}
    vn, en = mc_normal_var(df, w, conf=0.95, horizon=1, n_sim=800, seed=1)
    assert np.isfinite(vn) and np.isfinite(en)
    vt, et, nu2 = mc_student_t_var(df, w, conf=0.95, horizon=1, n_sim=800, seed=1)
    assert np.isfinite(vt) and np.isfinite(nu2)

    ps = parametric_var_sqrt10(r, 0.95)
    assert np.isfinite(ps)

    all1 = compute_all_var(df, w, conf=0.95, horizon=1, n_sim=1000)
    assert isinstance(all1, VaRResult)
    s = all1.to_series()
    assert "Historical VaR" in s.index

    all10 = compute_all_var(df, w, conf=0.95, horizon=10, n_sim=800)
    assert np.isfinite(all10.sqrt10_param_var)

    for method in ("historical", "parametric"):
        ts = compute_var_timeseries(df, w, conf=0.95, method=method, window=80)
        assert len(ts) > 0


def test_compute_var_timeseries_mc(monkeypatch):
    from src.risk import var as varmod
    from src.risk.var import compute_var_timeseries

    orig_n, orig_t = varmod.mc_normal_var, varmod.mc_student_t_var

    def wrap_n(*a, **kw):
        kw["n_sim"] = min(kw.get("n_sim", 10_000), 400)
        return orig_n(*a, **kw)

    def wrap_t(*a, **kw):
        kw["n_sim"] = min(kw.get("n_sim", 10_000), 400)
        return orig_t(*a, **kw)

    monkeypatch.setattr(varmod, "mc_normal_var", wrap_n)
    monkeypatch.setattr(varmod, "mc_student_t_var", wrap_t)
    idx = pd.bdate_range("2019-01-01", periods=200)
    rng = np.random.default_rng(21)
    df = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 2)), index=idx, columns=["SBER", "NVDA"])
    w = {"SBER": 0.5, "NVDA": 0.5}
    ts_n = compute_var_timeseries(df, w, conf=0.95, method="mc_normal", window=90)
    ts_t = compute_var_timeseries(df, w, conf=0.95, method="mc_t", window=90)
    assert len(ts_n) > 0 and len(ts_t) > 0


# --- Attribution -----------------------------------------------------------
def test_fx_decomposition_and_limit():
    from src.risk.attribution import (
        add_fx_decomposition,
        check_var_limit,
        compute_component_var,
        full_attribution,
    )

    idx = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(10)
    rets = pd.DataFrame(
        {
            "SBER": rng.normal(0, 0.012, len(idx)),
            "NVDA": rng.normal(0, 0.011, len(idx)),
        },
        index=idx,
    )
    fx = rng.normal(0, 0.004, len(idx))
    usdrub = pd.Series(fx, index=idx)
    w = {"SBER": 0.5, "NVDA": 0.5}
    attr = compute_component_var(rets, w, 0.95)
    attr2 = add_fx_decomposition(attr, rets, usdrub)
    dfx = attr2.to_dataframe()
    assert "fx_component" in dfx.columns

    ok = check_var_limit(0.001, 0.02, attr2)
    assert ok["breach"] is False

    breach = check_var_limit(0.5, 0.001, attr2)
    assert breach["breach"] is True
    assert breach.get("top_contributor") is not None

    fa, lc = full_attribution(rets, w, usdrub, conf=0.95, var_limit=0.02)
    assert "breach" in lc


# --- Kupiec ----------------------------------------------------------------
def test_kupiec_edges_and_run_all():
    from src.backtest.kupiec import kupiec_pof, kupiec_results_to_df, run_all_kupiec_tests

    z = kupiec_pof(np.zeros(500, dtype=int), 0.95, "all_zero")
    assert z.n_exceed == 0

    z2 = kupiec_pof(np.ones(50, dtype=int), 0.95, "all_one")
    assert z2.n_exceed == 50

    pnl = np.random.default_rng(11).normal(0, 0.01, 200)
    var_h = np.abs(pnl) * 0.3 + 0.001
    var_p = var_h * 1.05
    res = run_all_kupiec_tests(
        pnl,
        {"historical": var_h, "parametric": var_p},
        confidence=0.95,
    )
    assert len(res) == 2
    tbl = kupiec_results_to_df(res)
    assert "LR_uc" in tbl.columns


# --- Runner ----------------------------------------------------------------
def test_run_backtest_synthetic():
    from src.backtest.runner import run_backtest, run_all_backtests

    idx = pd.bdate_range("2019-01-01", "2020-03-20", freq="B")
    rng = np.random.default_rng(12)
    rets = pd.DataFrame(
        rng.normal(0, 0.01, (len(idx), 3)),
        index=idx,
        columns=["SBER", "LKOH", "NVDA"],
    )
    w = {"SBER": 0.34, "LKOH": 0.33, "NVDA": 0.33}
    bt = run_backtest(
        rets,
        w,
        period="COVID_2020",
        conf=0.95,
        methods=["historical", "parametric"],
        n_sim=500,
    )
    assert not bt.portfolio_pnl.empty
    assert "historical" in bt.var_series
    assert not bt.kupiec_table.empty

    all_bt = run_all_backtests(rets, w, conf=0.95)
    assert "COVID_2020" in all_bt


def test_run_all_backtests_handles_failure(monkeypatch):
    from src.backtest import runner as rmod

    orig = rmod.run_backtest

    def _fail(*a, **kw):
        period = kw.get("period", a[2] if len(a) > 2 else None)
        if period == "RateHike_2022":
            raise RuntimeError("simulated")
        return orig(*a, **kw)

    monkeypatch.setattr(rmod, "run_backtest", _fail)
    idx = pd.bdate_range("2019-01-01", "2020-03-20", freq="B")
    rng = np.random.default_rng(13)
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 3)), index=idx, columns=["SBER", "LKOH", "NVDA"])
    w = {"SBER": 0.34, "LKOH": 0.33, "NVDA": 0.33}
    out = rmod.run_all_backtests(rets, w, conf=0.95)
    assert "COVID_2020" in out


def test_kupiec_result_summary():
    from src.backtest.kupiec import kupiec_pof

    exc = np.zeros(400, dtype=int)
    exc[:20] = 1
    res = kupiec_pof(exc, 0.95, "m")
    s = res.summary()
    assert "LR_uc" in s and "m" in s


# --- Repository -----------------------------------------------------------
def test_repository_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKPULSE_DB", str(tmp_path / "t.db"))
    from src.db.repository import (
        get_db,
        init_db,
        load_close_pivot,
        load_macro,
        load_prices,
        upsert_macro,
        upsert_prices,
    )

    db = tmp_path / "t.db"
    init_db(db)
    rows = pd.DataFrame(
        {
            "ticker": ["SBER", "SBER"],
            "date": ["2020-01-02", "2020-01-03"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1e6, 1e6],
            "market": ["MOEX", "MOEX"],
            "currency": ["RUB", "RUB"],
        }
    )
    with get_db(db) as conn:
        upsert_prices(rows, conn)
    with get_db(db) as conn:
        ld = load_prices(["SBER"], "2020-01-01", "2020-06-01", conn)
        assert len(ld) == 2
        pv = load_close_pivot(["SBER"], "2020-01-01", "2020-06-01", conn)
        assert "SBER" in pv.columns

    m = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    with get_db(db) as conn:
        upsert_macro("FEDFUNDS", m, "TEST", conn)
    with get_db(db) as conn:
        s = load_macro("FEDFUNDS", "2020-01-01", "2020-06-01", conn)
        assert len(s) == 2


def test_db_path_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKPULSE_DB", str(tmp_path / "env.db"))
    from src.db import repository as rep

    assert rep._db_path() == tmp_path / "env.db"


def test_get_db_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKPULSE_DB", str(tmp_path / "r.db"))
    from src.db.repository import get_db, init_db

    init_db(tmp_path / "r.db")
    with pytest.raises(ValueError):
        with get_db(tmp_path / "r.db") as conn:
            conn.execute("INSERT INTO price_ohlcv VALUES ('X','2099-01-01',1,1,1,1,1,'M','R')")
            raise ValueError("abort")
    with sqlite3.connect(tmp_path / "r.db") as c:
        cur = c.execute("SELECT COUNT(*) FROM price_ohlcv WHERE ticker='X'")
        assert cur.fetchone()[0] == 0


# --- Macro -----------------------------------------------------------------
def test_cbr_fallback():
    from src.ingestion.macro_fetcher import _cbr_rate_fallback

    df = _cbr_rate_fallback("2020-01-01", "2020-06-30")
    assert not df.empty
    assert df["value"].notna().all()


# --- Price fetcher ---------------------------------------------------------
def test_fetch_moex_security_mocked():
    from unittest.mock import patch

    import requests

    from src.ingestion.price_fetcher import _fetch_moex_security

    fake = [
        {
            "TRADEDATE": "2020-01-10",
            "OPEN": 100,
            "HIGH": 101,
            "LOW": 99,
            "CLOSE": 100.5,
            "VOLUME": 1000,
        }
    ]
    with patch("src.ingestion.price_fetcher.apimoex.get_board_history", return_value=fake):
        df = _fetch_moex_security(requests.Session(), "SBER", "2020-01-01", "2020-02-01")
    assert df.iloc[0]["ticker"] == "SBER"
    assert df.iloc[0]["close"] == 100.5


def test_fetch_moex_empty_mocked():
    from unittest.mock import patch

    import requests

    from src.ingestion.price_fetcher import _fetch_moex_security

    with patch("src.ingestion.price_fetcher.apimoex.get_board_history", return_value=[]):
        df = _fetch_moex_security(requests.Session(), "SBER", "2020-01-01", "2020-02-01")
    assert df.empty


def test_run_update_triggers_fetch_when_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKPULSE_DB", str(tmp_path / "u.db"))
    from src.db.repository import init_db

    init_db(tmp_path / "u.db")
    with patch("src.db.repository.load_prices", return_value=pd.DataFrame()):
        with patch("src.ingestion.price_fetcher.run_fetch") as rf:
            from src.ingestion.price_fetcher import run_update

            run_update()
    rf.assert_called_once()


# --- run_pipeline CLI ------------------------------------------------------
def test_run_pipeline_main_compute_mocked(monkeypatch):
    import run_pipeline as rp

    monkeypatch.setattr(rp, "step_compute", lambda **k: None)
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--compute"])
    rp.main()


def test_run_pipeline_main_no_flags_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_pipeline"])
    import run_pipeline as rp

    rp.main()
    cap = capsys.readouterr()
    text = (cap.out + cap.err).lower()
    assert "usage" in text or "fetch" in text or "riskpulse" in text


def test_run_pipeline_dashboard_mock_subprocess(monkeypatch):
    import subprocess

    import run_pipeline as rp

    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--dashboard"])
    with patch.object(subprocess, "run", return_value=MagicMock(returncode=0)):
        rp.main()


def test_run_pipeline_all_mocked(monkeypatch):
    import run_pipeline as rp

    monkeypatch.setattr(rp, "step_fetch_prices", lambda: None)
    monkeypatch.setattr(rp, "step_fetch_macro", lambda k: None)
    monkeypatch.setattr(rp, "step_compute", lambda **k: None)
    monkeypatch.setattr(rp, "step_backtest", lambda **k: None)
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--all", "--fred-key", "dummy"])
    rp.main()


def test_run_pipeline_fetch_and_update_mocked(monkeypatch):
    import run_pipeline as rp

    monkeypatch.setattr(rp, "step_fetch_prices", lambda: None)
    monkeypatch.setattr(rp, "step_fetch_macro", lambda k: None)
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--fetch", "--fred-key", "k"])
    rp.main()

    with patch("src.ingestion.price_fetcher.run_update", lambda: None):
        monkeypatch.setattr(sys, "argv", ["run_pipeline", "--update"])
        rp.main()


def test_macro_fetch_fred_series_mocked():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    ser = pd.Series([1.25, 1.26], index=idx)
    with patch("fredapi.Fred") as MF:
        MF.return_value.get_series.return_value = ser
        from src.ingestion.macro_fetcher import fetch_fred_series

        out = fetch_fred_series("FEDFUNDS", "2020-01-01", "2020-02-01", "dummy-key")
    assert len(out) == 2
    assert "value" in out.columns


def test_macro_run_fetch_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKPULSE_DB", str(tmp_path / "mf.db"))
    from src.db.repository import init_db

    init_db(tmp_path / "mf.db")
    idx = pd.to_datetime(["2020-01-02"])
    fred_df = pd.DataFrame({"value": [2.0]}, index=idx)
    cbr_df = pd.DataFrame({"value": [7.5]}, index=idx)
    with patch("src.ingestion.macro_fetcher.fetch_fred_series", return_value=fred_df):
        with patch("src.ingestion.macro_fetcher.fetch_cbr_key_rate", return_value=cbr_df):
            from src.ingestion.macro_fetcher import run_fetch

            run_fetch("k", start="2020-01-01", end="2020-02-15")


def test_fetch_usdrub_mocked():
    from unittest.mock import MagicMock

    from src.ingestion.price_fetcher import _fetch_usdrub

    sess = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {
        "history": {
            "columns": ["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "WAPRICE"],
            "data": [["2020-01-10", 60.0, 61.0, 59.0, 60.5, None]],
        }
    }
    resp.raise_for_status = MagicMock()
    sess.get.return_value = resp
    df = _fetch_usdrub(sess, "2020-01-01", "2020-02-01")
    assert not df.empty and df.iloc[0]["ticker"] == "USDRUB"


def test_fetch_usdrub_no_rows():
    from unittest.mock import MagicMock

    from src.ingestion.price_fetcher import _fetch_usdrub

    sess = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"history": {"columns": ["TRADEDATE", "CLOSE"], "data": []}}
    resp.raise_for_status = MagicMock()
    sess.get.return_value = resp
    df = _fetch_usdrub(sess, "2020-01-01", "2020-02-01")
    assert df.empty


def test_run_fetch_prices_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKPULSE_DB", str(tmp_path / "pf.db"))
    from src.db.repository import init_db

    init_db(tmp_path / "pf.db")

    def _row(ticker: str, market: str, cur: str) -> dict:
        return {
            "ticker": ticker,
            "date": "2020-01-10",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1e6,
            "market": market,
            "currency": cur,
        }

    moex = pd.DataFrame([_row("SBER", "MOEX", "RUB")])
    us = pd.DataFrame([_row("NVDA", "US", "USD"), _row("MSFT", "US", "USD")])
    with patch("src.ingestion.price_fetcher.fetch_moex_all", return_value=moex):
        with patch("src.ingestion.price_fetcher.fetch_us_all", return_value=us):
            from src.ingestion.price_fetcher import run_fetch

            run_fetch(start="2020-01-01", end="2020-02-01")


def test_build_hypothetical_shocks_without_brent():
    from src.risk.stress import build_hypothetical_shocks

    idx = pd.bdate_range("2019-01-01", periods=200)
    rng = np.random.default_rng(23)
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 4)), index=idx, columns=["SBER", "LKOH", "GAZP", "NVDA"])
    fx = pd.Series(rng.normal(0, 0.005, len(idx)), index=idx)
    shocks = build_hypothetical_shocks(rets, fx, brent_returns=None)
    assert len(shocks) == 3


def test_fetch_imoex_mocked():
    from unittest.mock import patch

    import requests

    from src.ingestion.price_fetcher import _fetch_imoex

    row = {
        "TRADEDATE": "2020-01-10",
        "OPEN": 2700,
        "HIGH": 2750,
        "LOW": 2680,
        "CLOSE": 2720,
        "VOLUME": 0,
    }
    with patch("src.ingestion.price_fetcher.apimoex.get_board_history", return_value=[row]):
        df = _fetch_imoex(requests.Session(), "2020-01-01", "2020-02-01")
    assert df.iloc[0]["ticker"] == "IMOEX"


def _seed_synthetic_prices_db(db_path: Path) -> None:
    from src.db.repository import get_db, init_db, upsert_prices

    init_db(db_path)
    di = pd.bdate_range("2019-01-01", periods=320)
    rng = np.random.default_rng(0)
    tickers_info = [
        ("SBER", "MOEX", "RUB", 250.0),
        ("LKOH", "MOEX", "RUB", 4000.0),
        ("GAZP", "MOEX", "RUB", 160.0),
        ("NVDA", "US", "USD", 500.0),
        ("MSFT", "US", "USD", 300.0),
        ("AAPL", "US", "USD", 180.0),
        ("IMOEX", "MOEX", "RUB", 2800.0),
        ("SPY", "US", "USD", 400.0),
        ("USDRUB", "MOEX", "RUB", 65.0),
    ]
    rows: list[dict] = []
    for i, d in enumerate(di):
        for t, mkt, cur, base in tickers_info:
            noise = 1.0 + 0.002 * rng.standard_normal()
            c = float(base * (1.0 + 0.0003 * i) * noise)
            ds = str(d.date())
            rows.append(
                {
                    "ticker": t,
                    "date": ds,
                    "open": c,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 1e6,
                    "market": mkt,
                    "currency": cur,
                }
            )
    df = pd.DataFrame(rows)
    with get_db(db_path) as conn:
        upsert_prices(df, conn)


def test_run_pipeline_step_compute_integration(tmp_path, monkeypatch):
    from src.risk import var as V

    orig = V.compute_all_var

    def _fast(*a, **kw):
        return orig(*a, **{**kw, "n_sim": min(int(kw.get("n_sim", 5000)), 800)})

    monkeypatch.setattr(V, "compute_all_var", _fast)
    db = tmp_path / "rp.sqlite"
    monkeypatch.setenv("RISKPULSE_DB", str(db))
    _seed_synthetic_prices_db(db)
    import run_pipeline as rp

    rp.step_compute(False)


def test_run_pipeline_step_compute_prints(capsys, tmp_path, monkeypatch):
    from src.risk import var as V

    real = V.compute_all_var
    monkeypatch.setattr(V, "compute_all_var", lambda *a, **kw: real(*a, **{**kw, "n_sim": 600}))
    db = tmp_path / "rp2.sqlite"
    monkeypatch.setenv("RISKPULSE_DB", str(db))
    _seed_synthetic_prices_db(db)
    import run_pipeline as rp

    rp.step_compute(True)
    out = capsys.readouterr().out
    assert "RISKPULSE" in out or "VaR" in out


def test_run_pipeline_step_backtest_integration(capsys, tmp_path, monkeypatch):
    from src.backtest import runner as br

    real_bt = br.run_backtest

    def fast_bt(*a, **kw):
        kw = {**kw, "methods": ["historical", "parametric"], "n_sim": 400}
        return real_bt(*a, **kw)

    monkeypatch.setattr(br, "run_backtest", fast_bt)
    db = tmp_path / "rp3.sqlite"
    monkeypatch.setenv("RISKPULSE_DB", str(db))
    _seed_synthetic_prices_db(db)
    import run_pipeline as rp

    rp.step_backtest(0.95)
    out = capsys.readouterr().out
    assert "BACKTEST" in out or "LR_uc" in out


def test_fetch_cbr_key_rate_uses_fallback_on_csv_error():
    with patch("src.ingestion.macro_fetcher.pd.read_csv", side_effect=RuntimeError("offline")):
        from src.ingestion.macro_fetcher import fetch_cbr_key_rate

        df = fetch_cbr_key_rate("2020-01-01", "2020-06-01")
    assert not df.empty


def test_fetch_moex_all_mocked():
    from src.ingestion.price_fetcher import fetch_moex_all

    def _ohlc_row(ticker: str) -> dict:
        return {
            "date": "2020-01-10",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "volume": 1.0,
            "ticker": ticker,
            "market": "MOEX",
            "currency": "RUB",
        }

    sber = pd.DataFrame([_ohlc_row("SBER")])
    lkoh = pd.DataFrame([_ohlc_row("LKOH")])
    gazp = pd.DataFrame([_ohlc_row("GAZP")])
    imoex = pd.DataFrame([_ohlc_row("IMOEX")])
    usd = pd.DataFrame([_ohlc_row("USDRUB")])

    with patch("src.ingestion.price_fetcher._fetch_moex_security", side_effect=[sber, lkoh, gazp]):
        with patch("src.ingestion.price_fetcher._fetch_imoex", return_value=imoex):
            with patch("src.ingestion.price_fetcher._fetch_usdrub", return_value=usd):
                df = fetch_moex_all("2020-01-01", "2020-02-01")
    assert len(df) >= 5


def test_capm_load_french_download_mocked(tmp_path, monkeypatch):
    import io
    import zipfile

    from src.risk import capm

    capm._french_cache = None
    csv_lines = "\n".join(
        [
            "junk header",
            "20200102,10,20,30,40",
            "20200103,11,21,31,41",
        ]
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Research_Data_Factors_daily.CSV", csv_lines)
    buf.seek(0)

    class Resp:
        content = buf.getvalue()

        def raise_for_status(self):
            pass

    p = tmp_path / "ff.parquet"
    monkeypatch.setattr(capm.requests, "get", lambda url, timeout=60: Resp())
    out = capm.load_french_factors(p)
    assert not out.empty
    capm._french_cache = None
