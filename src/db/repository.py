"""SQLite: init, upserts, контекст get_db()."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd

_DEFAULT_DB = Path(__file__).parents[2] / "data" / "riskpulse.db"
_SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def _db_path() -> Path:
    env = os.environ.get("RISKPULSE_DB")
    return Path(env) if env else _DEFAULT_DB


def init_db(path: Path | None = None) -> None:
    """Create tables if they don't exist."""
    db = path or _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(_SCHEMA_FILE.read_text())
        conn.commit()


@contextmanager
def get_db(path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    db = path or _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()



def upsert_prices(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """df must have columns: ticker, date, open, high, low, close, volume, market, currency"""
    rows = df[["ticker", "date", "open", "high", "low", "close", "volume", "market", "currency"]].values.tolist()
    conn.executemany(
        """
        INSERT INTO price_ohlcv (ticker, date, open, high, low, close, volume, market, currency)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume
        """,
        rows,
    )


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    placeholders = ",".join("?" * len(tickers))
    sql = f"""
        SELECT ticker, date, close, market, currency
        FROM price_ohlcv
        WHERE ticker IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY ticker, date
    """
    df = pd.read_sql_query(sql, conn, params=[*tickers, start, end])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_close_pivot(
    tickers: list[str],
    start: str,
    end: str,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """Returns wide DataFrame: index=date, columns=tickers, values=close."""
    df = load_prices(tickers, start, end, conn)
    return df.pivot(index="date", columns="ticker", values="close").sort_index()



def upsert_macro(series_id: str, df: pd.DataFrame, source: str, conn: sqlite3.Connection) -> None:
    """df: index=date (datetime), single column 'value'."""
    rows = [(series_id, str(d.date()), float(v), source) for d, v in df["value"].items()]
    conn.executemany(
        """
        INSERT INTO macro_series (series_id, date, value, source)
        VALUES (?,?,?,?)
        ON CONFLICT(series_id, date) DO UPDATE SET value=excluded.value
        """,
        rows,
    )


def load_macro(series_id: str, start: str, end: str, conn: sqlite3.Connection) -> pd.Series:
    sql = """
        SELECT date, value FROM macro_series
        WHERE series_id=? AND date>=? AND date<=?
        ORDER BY date
    """
    df = pd.read_sql_query(sql, conn, params=[series_id, start, end])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].rename(series_id)
