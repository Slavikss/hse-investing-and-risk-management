"""Tests for src/analysis — portfolio comparison and hyperopt."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio_comparison import (
    PORTFOLIO_VARIANTS,
    PortfolioMetrics,
    _max_drawdown,
    _ols_beta,
    benchmark_metrics,
    compare_all,
    compute_metrics,
    rolling_performance,
)
from src.analysis.hyperopt import (
    LambdaOptResult,
    WeightOptResult,
    WindowOptResult,
    optimize_capm_window,
    optimize_lambda,
    optimize_weights,
    run_all_optimisations,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_returns(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    data = {t: rng.normal(0.0005, 0.01, n) for t in ["SBER", "LKOH", "GAZP", "NVDA", "MSFT", "AAPL"]}
    return pd.DataFrame(data, index=idx)


def _make_bench(n: int = 300, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(rng.normal(0.0003, 0.008, n), index=idx, name="bench")


# ─── portfolio_comparison ─────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_flat(self):
        assert _max_drawdown(np.zeros(50)) == pytest.approx(0.0)

    def test_monotone_up(self):
        r = np.full(50, 0.01)
        assert _max_drawdown(r) >= -1e-9

    def test_single_drop(self):
        r = np.array([0.1, 0.0, -0.5, 0.1, 0.1])
        dd = _max_drawdown(r)
        assert dd < 0.0

    def test_returns_float(self):
        r = np.random.default_rng(0).normal(0, 0.01, 200)
        result = _max_drawdown(r)
        assert isinstance(result, float)
        assert result <= 0.0


class TestOlsBeta:
    def test_known_beta(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2022-01-03", periods=200)
        x = pd.Series(rng.normal(0, 0.01, 200), index=idx)
        y = 1.5 * x + rng.normal(0, 0.002, 200)
        y = pd.Series(y, index=idx)
        beta = _ols_beta(y, x)
        assert beta is not None
        assert abs(beta - 1.5) < 0.1

    def test_short_series_returns_none(self):
        idx = pd.bdate_range("2022-01-03", periods=20)
        x = pd.Series(np.ones(20) * 0.01, index=idx)
        y = pd.Series(np.ones(20) * 0.01, index=idx)
        assert _ols_beta(y, x) is None


class TestComputeMetrics:
    def setup_method(self):
        self.rets = _make_returns()
        self.spy  = _make_bench()
        self.weights = {"SBER": 0.2, "LKOH": 0.2, "GAZP": 0.1, "NVDA": 0.2, "MSFT": 0.15, "AAPL": 0.15}

    def test_returns_dataclass(self):
        m = compute_metrics("Test", self.weights, self.rets, n_sim=200)
        assert isinstance(m, PortfolioMetrics)

    def test_sharpe_finite(self):
        m = compute_metrics("Test", self.weights, self.rets, n_sim=200)
        assert np.isfinite(m.sharpe)

    def test_var_positive(self):
        m = compute_metrics("Test", self.weights, self.rets, n_sim=200)
        assert m.var_95_1d > 0
        assert m.es_95_1d  > 0
        assert m.var_99_1d > 0
        assert m.var_95_10d > 0

    def test_beta_spy_computed(self):
        m = compute_metrics("Test", self.weights, self.rets, bench_spy=self.spy, n_sim=200)
        assert m.beta_spy is not None
        assert np.isfinite(m.beta_spy)

    def test_max_drawdown_nonpositive(self):
        m = compute_metrics("Test", self.weights, self.rets, n_sim=200)
        assert m.max_drawdown <= 0.0

    def test_to_series(self):
        m = compute_metrics("Test", self.weights, self.rets, n_sim=200)
        s = m.to_series()
        assert "Портфель" in s.index
        assert "Sharpe" in s.index

    def test_weights_normalised(self):
        w_unnorm = {"SBER": 3, "LKOH": 3, "GAZP": 2, "NVDA": 2, "MSFT": 2, "AAPL": 2}
        m = compute_metrics("Norm", w_unnorm, self.rets, n_sim=200)
        assert abs(sum(m.weights.values()) - 1.0) < 1e-8

    def test_missing_ticker_ignored(self):
        w = {"SBER": 0.5, "MISSING": 0.5}
        m = compute_metrics("Partial", w, self.rets, n_sim=200)
        assert "MISSING" not in m.weights

    def test_all_missing_raises(self):
        with pytest.raises(ValueError):
            compute_metrics("Bad", {"MISSING": 1.0}, self.rets, n_sim=200)


class TestBenchmarkMetrics:
    def test_returns_series(self):
        bench = _make_bench()
        s = benchmark_metrics("SPY", bench)
        assert "Ann. Return" in s.index
        assert "Sharpe" in s.index
        assert s["β(SPY)"] == "1.000"

    def test_imoex_label(self):
        bench = _make_bench()
        s = benchmark_metrics("IMOEX", bench)
        assert s["β(IMOEX)"] == "1.000"


class TestCompareAll:
    def test_returns_dataframe(self):
        rets = _make_returns()
        df = compare_all(rets, n_sim=200)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= len(PORTFOLIO_VARIANTS)

    def test_includes_benchmarks(self):
        rets = _make_returns()
        spy  = _make_bench()
        imoex = _make_bench(seed=11)
        df = compare_all(rets, spy_ret=spy, imoex_ret=imoex, n_sim=200)
        names = df["Портфель"].tolist()
        assert "SPY" in names
        assert "IMOEX (RUB)" in names

    def test_extra_portfolios(self):
        rets = _make_returns()
        extra = {"Custom": {"SBER": 0.5, "NVDA": 0.5}}
        df = compare_all(rets, extra_portfolios=extra, n_sim=200)
        assert "Custom" in df["Портфель"].tolist()


class TestRollingPerformance:
    def test_returns_dataframe(self):
        rets = _make_returns(400)
        w = {"SBER": 0.3, "NVDA": 0.3, "MSFT": 0.2, "AAPL": 0.2}
        df = rolling_performance(w, rets, window=63)
        assert isinstance(df, pd.DataFrame)
        assert "sharpe" in df.columns
        assert "var_ewma" in df.columns
        assert len(df) == len(rets) - 63

    def test_length_with_window(self):
        rets = _make_returns(200)
        w = {"SBER": 0.5, "NVDA": 0.5}
        df = rolling_performance(w, rets, window=50)
        assert len(df) == 200 - 50


# ─── hyperopt ─────────────────────────────────────────────────────────────────

class TestOptimiseLambda:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.r = rng.normal(0, 0.01, 200)

    def test_returns_dataclass(self):
        res = optimize_lambda(self.r, conf=0.95, lambdas=[0.90, 0.94, 0.97], min_train=40)
        assert isinstance(res, LambdaOptResult)

    def test_best_lambda_in_grid(self):
        lams = [0.90, 0.93, 0.97]
        res = optimize_lambda(self.r, conf=0.95, lambdas=lams, min_train=40)
        assert res.best_lambda in lams

    def test_target_rate(self):
        res = optimize_lambda(self.r, conf=0.95, lambdas=[0.94], min_train=40)
        assert abs(res.target_rate - 0.05) < 1e-9

    def test_grid_has_all_lambdas(self):
        lams = [0.88, 0.92, 0.96]
        res = optimize_lambda(self.r, conf=0.95, lambdas=lams, min_train=40)
        assert len(res.grid) == len(lams)

    def test_default_lambdas(self):
        res = optimize_lambda(self.r, conf=0.95, min_train=40)
        assert len(res.grid) > 5

    def test_actual_rate_between_0_1(self):
        res = optimize_lambda(self.r, conf=0.95, lambdas=[0.94], min_train=40)
        assert 0.0 <= res.best_actual_rate <= 1.0


class TestOptimiseCapmWindow:
    def setup_method(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-02", periods=500)
        self.x = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        self.y = 1.2 * self.x + rng.normal(0, 0.003, 500)
        self.y = pd.Series(self.y.values, index=idx)

    def test_returns_dataclass(self):
        res = optimize_capm_window(self.y, self.x, windows=[60, 120])
        assert isinstance(res, WindowOptResult)

    def test_best_window_in_grid(self):
        windows = [60, 120, 180]
        res = optimize_capm_window(self.y, self.x, windows=windows)
        assert res.best_window in windows

    def test_avg_r2_positive(self):
        res = optimize_capm_window(self.y, self.x, windows=[60, 90])
        assert res.best_avg_r2 > 0.0

    def test_short_series_nan(self):
        idx = pd.bdate_range("2020-01-02", periods=50)
        x = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50), index=idx)
        y = pd.Series(np.random.default_rng(1).normal(0, 0.01, 50), index=idx)
        res = optimize_capm_window(y, x, windows=[100, 200])
        assert res.grid["avg_r2"].isna().all()


class TestOptimiseWeights:
    def setup_method(self):
        self.rets = _make_returns(400)

    def test_sharpe_returns_dataclass(self):
        res = optimize_weights(self.rets, objective="sharpe")
        assert isinstance(res, WeightOptResult)

    def test_weights_sum_to_one(self):
        for obj in ("sharpe", "min_var", "min_vol"):
            res = optimize_weights(self.rets, objective=obj)
            total = sum(res.optimal_weights.values())
            assert abs(total - 1.0) < 1e-4, f"{obj}: sum={total}"

    def test_weights_within_bounds(self):
        res = optimize_weights(self.rets, objective="sharpe", min_w=0.05, max_w=0.50)
        for w in res.optimal_weights.values():
            assert w >= 0.04  # small tolerance
            assert w <= 0.51

    def test_all_tickers_present(self):
        res = optimize_weights(self.rets, objective="min_vol")
        assert set(res.optimal_weights) == set(self.rets.columns)

    def test_unknown_objective_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            optimize_weights(self.rets, objective="magic")

    def test_calmar_objective(self):
        res = optimize_weights(self.rets, objective="calmar")
        assert isinstance(res, WeightOptResult)

    def test_min_es_objective(self):
        res = optimize_weights(self.rets, objective="min_es")
        assert isinstance(res, WeightOptResult)


class TestRunAllOptimisations:
    def test_runs_without_error(self):
        rets = _make_returns(300)
        rng = np.random.default_rng(42)
        port_r = rng.normal(0, 0.01, 300)
        out = run_all_optimisations(rets, port_r, conf=0.95)
        assert "lambda" in out
        assert "weights_sharpe" in out

    def test_with_spy(self):
        rets  = _make_returns(300)
        rng   = np.random.default_rng(42)
        port_r = rng.normal(0, 0.01, 300)
        spy   = _make_bench(300)
        out = run_all_optimisations(rets, port_r, conf=0.95, spy_ret=spy, main_asset="SBER")
        assert "capm_window" in out
