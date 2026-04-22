"""Макро через FRED (+ MOEX для ставки ЦБ при необходимости)."""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date

import pandas as pd
import requests

from src.db.repository import get_db, init_db, upsert_macro

logger = logging.getLogger(__name__)

DEFAULT_START = "2018-01-01"

FRED_SERIES = {
    "FEDFUNDS":     "Fed Funds Rate (%)",
    "CPIAUCSL":     "CPI USA",
    "DTWEXBGS":     "USD Trade-Weighted Index (DXY-proxy)",
    "DCOILBRENTEU": "Brent Crude Oil (USD/bbl)",
}

CBR_MOEX_URL = (
    "https://iss.moex.com/iss/statistics/engines/currency/markets/selt/"
    "rates.json?iss.meta=off"
)



def fetch_fred_series(series_id: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Возвращает DataFrame с колонкой 'value' и DatetimeIndex."""
    from fredapi import Fred  # type: ignore
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s = s.dropna()
    df = s.reset_index()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")



def fetch_cbr_key_rate(start: str, end: str) -> pd.DataFrame:
    """
    Ключевая ставка ЦБ РФ — исторический ряд из открытого источника.
    Используем статический CSV от ЦБ РФ (XML → pandas).
    """
    # ЦБ публикует XML со ставками; используем простой CSV из публичного источника
    url = "https://raw.githubusercontent.com/s-kganz/cbr_rate/main/cbr_key_rate.csv"
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col  = [c for c in df.columns if "date" in c or "дата" in c][0]
        value_col = [c for c in df.columns if "rate" in c or "ставка" in c or "value" in c][0]
        df = df.rename(columns={date_col: "date", value_col: "value"})
        df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df = df.set_index("date").sort_index()
        # forward-fill в daily частоту
        idx = pd.date_range(start, end, freq="B")
        df = df["value"].reindex(idx, method="ffill").dropna()
        result = df.reset_index()
        result.columns = ["date", "value"]
        return result.set_index("date")
    except Exception as e:
        logger.warning("CBR rate fetch failed (%s); using fallback hardcoded series", e)
        return _cbr_rate_fallback(start, end)


def _cbr_rate_fallback(start: str, end: str) -> pd.DataFrame:
    """
    Хардкодим ключевые точки ставки ЦБ РФ (% годовых) — достаточно для rolling CAPM.
    Последнее обновление: Апрель 2025.
    """
    breakpoints = [
        ("2018-01-01", 7.75),
        ("2018-03-26", 7.25),
        ("2018-09-17", 7.50),
        ("2018-12-17", 7.75),
        ("2019-06-17", 7.50),
        ("2019-07-29", 7.25),
        ("2019-09-09", 7.00),
        ("2019-10-28", 6.50),
        ("2019-12-16", 6.25),
        ("2020-02-10", 6.00),
        ("2020-04-27", 5.50),
        ("2020-06-22", 4.50),
        ("2020-07-27", 4.25),
        ("2021-03-19", 4.50),
        ("2021-04-23", 5.00),
        ("2021-06-11", 5.50),
        ("2021-07-23", 6.50),
        ("2021-09-10", 6.75),
        ("2021-10-22", 7.50),
        ("2021-12-17", 8.50),
        ("2022-02-28", 20.00),
        ("2022-04-11", 17.00),
        ("2022-05-04", 14.00),
        ("2022-05-27", 11.00),
        ("2022-06-10", 9.50),
        ("2022-07-25", 8.00),
        ("2022-09-16", 7.50),
        ("2023-07-21", 8.50),
        ("2023-08-15", 12.00),
        ("2023-09-15", 13.00),
        ("2023-10-27", 15.00),
        ("2023-12-15", 16.00),
        ("2024-07-26", 18.00),
        ("2024-09-13", 19.00),
        ("2024-10-25", 21.00),
        ("2024-12-20", 21.00),
        ("2025-02-14", 21.00),
    ]
    idx = pd.date_range(start, end, freq="B")
    s = pd.Series(dict(breakpoints), dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.reindex(idx, method="ffill").ffill().bfill()
    result = s.reset_index()
    result.columns = ["date", "value"]
    return result.set_index("date")



def run_fetch(fred_key: str, start: str = DEFAULT_START, end: str | None = None) -> None:
    if end is None:
        end = str(date.today())
    init_db()

    with get_db() as conn:
        # FRED series
        for series_id, desc in FRED_SERIES.items():
            logger.info("Fetching FRED: %s (%s)", series_id, desc)
            try:
                df = fetch_fred_series(series_id, start, end, fred_key)
                upsert_macro(series_id, df, "FRED", conn)
                logger.info("  → %d observations", len(df))
            except Exception as exc:
                logger.error("FRED %s failed: %s", series_id, exc)

        # ЦБ РФ
        logger.info("Fetching CBR key rate")
        try:
            df = fetch_cbr_key_rate(start, end)
            upsert_macro("CBRATE", df, "CBR", conn)
            logger.info("  → %d observations", len(df))
        except Exception as exc:
            logger.error("CBR rate failed: %s", exc)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fred-key", default=os.environ.get("FRED_API_KEY", ""))
    parser.add_argument("--start",   default=DEFAULT_START)
    parser.add_argument("--end",     default=None)
    args = parser.parse_args()
    if not args.fred_key:
        raise SystemExit("Нужен FRED API key: --fred-key KEY или переменная FRED_API_KEY")
    run_fetch(fred_key=args.fred_key, start=args.start, end=args.end)
