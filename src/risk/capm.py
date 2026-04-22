"""Rolling CAPM (RU/RUB vs IMOEX, Global/USD vs SPY) и FF3 (Ken French), OLS через NumPy."""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

WINDOW = 252
MIN_OBS = 60

RU_TICKERS     = ["SBER", "LKOH", "GAZP"]
GLOBAL_TICKERS = ["NVDA", "MSFT", "AAPL"]
FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def _annual_rate_to_daily(rate_pct: pd.Series) -> pd.Series:
    return np.log(1 + rate_pct / 100) / 252


def _add_constant_series(s: pd.Series, name: str = "x") -> pd.DataFrame:
    return pd.concat(
        [pd.Series(1.0, index=s.index, name="const"), s.rename(name)],
        axis=1,
    )


def _rolling_ols_single(
    y: pd.Series,
    X: pd.DataFrame,
    window: int = WINDOW,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    y = y.dropna()
    X = X.reindex(y.index).dropna(how="any")
    y = y.reindex(X.index)
    n = len(y)
    if n < window:
        return pd.DataFrame()

    rows: list[list[float]] = []
    idx_out: list = []
    Xv = X.values.astype(float)
    yv = y.values.astype(float)

    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        yw, Xw = yv[sl], Xv[sl]
        if not (np.isfinite(yw).all() and np.isfinite(Xw).all()):
            continue
        if yw.size < min_obs:
            continue
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        pred = Xw @ coef
        ss_res = float(np.sum((yw - pred) ** 2))
        ss_tot = float(np.sum((yw - np.mean(yw)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
        rows.append([float(coef[0]), float(coef[1]), r2])
        idx_out.append(y.index[i])

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=idx_out, columns=["alpha", "beta", "r_squared"])


def _rolling_ols_ff3(
    y: pd.Series,
    X: pd.DataFrame,
    window: int = WINDOW,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    y = y.dropna()
    X = X.reindex(y.index).dropna(how="any")
    y = y.reindex(X.index)
    n = len(y)
    if n < window:
        return pd.DataFrame()

    rows = []
    idx_out = []
    Xv = X.values.astype(float)
    yv = y.values.astype(float)

    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        yw, Xw = yv[sl], Xv[sl]
        if not (np.isfinite(yw).all() and np.isfinite(Xw).all()):
            continue
        if yw.size < min_obs:
            continue
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        pred = Xw @ coef
        ss_res = float(np.sum((yw - pred) ** 2))
        ss_tot = float(np.sum((yw - np.mean(yw)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
        rows.append([float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3]), r2])
        idx_out.append(y.index[i])

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        rows,
        index=idx_out,
        columns=["alpha", "beta", "beta_smb", "beta_hml", "r_squared"],
    )


_french_cache: pd.DataFrame | None = None


def load_french_factors(cache_path: Path | None = None) -> pd.DataFrame:
    global _french_cache
    if _french_cache is not None:
        return _french_cache

    if cache_path is None:
        cache_path = Path(__file__).parents[2] / "data" / "ff3_daily.parquet"

    if cache_path.exists():
        _french_cache = pd.read_parquet(cache_path)
        return _french_cache

    logger.info("Downloading Fama-French daily factors from Ken French site...")
    resp = requests.get(FRENCH_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw = z.read(csv_name).decode("latin-1")

    lines = raw.splitlines()
    data_lines = []
    for line in lines:
        parts = line.split(",")
        if len(parts) >= 5 and len(parts[0].strip()) == 8:
            try:
                int(parts[0].strip())
                data_lines.append(line)
            except ValueError:
                pass

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=["date", "mkt_rf", "smb", "hml", "rf"],
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    for col in ["mkt_rf", "smb", "hml", "rf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100
    df = df.dropna().set_index("date").sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    _french_cache = df
    return df


def rolling_capm_ru(
    prices_rub: pd.DataFrame,
    imoex: pd.Series,
    cbr_rate: pd.Series,
    window: int = WINDOW,
) -> pd.DataFrame:
    rets_assets = _log_returns(prices_rub)
    rets_imoex  = _log_returns(imoex.to_frame("IMOEX"))["IMOEX"]

    rf_daily = _annual_rate_to_daily(cbr_rate.reindex(rets_assets.index).ffill())

    mkt_excess = rets_imoex - rf_daily

    results = []
    for ticker in rets_assets.columns:
        if ticker not in RU_TICKERS:
            continue
        y = rets_assets[ticker] - rf_daily
        y, mkt = y.align(mkt_excess, join="inner")
        y = y.dropna()
        mkt = mkt.reindex(y.index).dropna()
        y = y.reindex(mkt.index)

        X = _add_constant_series(mkt.rename("mkt"), "mkt")
        params = _rolling_ols_single(y, X, window)
        params["ticker"] = ticker
        params = params.reset_index().rename(columns={"index": "date"})
        results.append(params)

    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    out["cluster"] = "RU"
    return out[["date", "ticker", "cluster", "alpha", "beta", "r_squared"]]


def rolling_capm_global(
    prices_usd: pd.DataFrame,
    spy: pd.Series,
    fed_rate: pd.Series,
    window: int = WINDOW,
) -> pd.DataFrame:
    rets_assets = _log_returns(prices_usd)
    rets_spy    = _log_returns(spy.to_frame("SPY"))["SPY"]
    rf_daily    = _annual_rate_to_daily(fed_rate.reindex(rets_assets.index).ffill())

    mkt_excess = rets_spy - rf_daily

    results = []
    for ticker in rets_assets.columns:
        if ticker not in GLOBAL_TICKERS:
            continue
        y = rets_assets[ticker] - rf_daily
        y, mkt = y.align(mkt_excess, join="inner")
        y = y.dropna()
        mkt = mkt.reindex(y.index).dropna()
        y = y.reindex(mkt.index)

        X = _add_constant_series(mkt.rename("mkt"), "mkt")
        params = _rolling_ols_single(y, X, window)
        params["ticker"] = ticker
        params = params.reset_index().rename(columns={"index": "date"})
        results.append(params)

    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    out["cluster"] = "Global"
    return out[["date", "ticker", "cluster", "alpha", "beta", "r_squared"]]


def rolling_ff3_global(
    prices_usd: pd.DataFrame,
    ff3: pd.DataFrame,
    window: int = WINDOW,
) -> pd.DataFrame:
    rets_assets = _log_returns(prices_usd)

    results = []
    for ticker in rets_assets.columns:
        if ticker not in GLOBAL_TICKERS:
            continue
        y = rets_assets[[ticker]].join(ff3, how="inner").dropna()
        excess = (y[ticker] - y["rf"]).rename("excess")
        X = pd.concat(
            [pd.Series(1.0, index=y.index, name="const"), y[["mkt_rf", "smb", "hml"]]],
            axis=1,
        )
        X.columns = ["const", "mkt", "smb", "hml"]

        p = _rolling_ols_ff3(excess, X, window=window, min_obs=MIN_OBS)
        p["ticker"] = ticker
        p = p.reset_index().rename(columns={"index": "date"})
        results.append(p)

    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    out["cluster"] = "Global_FF3"
    return out[["date", "ticker", "cluster", "alpha", "beta", "beta_smb", "beta_hml", "r_squared"]]


def latest_betas(capm_ru: pd.DataFrame, capm_global: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df in [capm_ru, capm_global]:
        if df.empty:
            continue
        latest = df.sort_values("date").groupby("ticker").last().reset_index()
        rows.append(latest)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
