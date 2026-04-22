"""MOEX (apimoex), USDRUB_TOM, yfinance; запись в SQLite."""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import apimoex
import pandas as pd
import requests
import yfinance as yf

from src.db.repository import get_db, init_db, upsert_prices

logger = logging.getLogger(__name__)


MOEX_TICKERS = ["SBER", "LKOH", "GAZP", "IMOEX"]  # IMOEX — бенчмарк
US_TICKERS   = ["NVDA", "MSFT", "AAPL", "SPY"]     # SPY — бенчмарк
FX_TICKER    = "USDRUB_TOM"

DEFAULT_START = "2018-01-01"



def _fetch_moex_security(
    session: requests.Session,
    ticker: str,
    start: str,
    end: str,
    board: str = "TQBR",
    market: str = "shares",
) -> pd.DataFrame:
    """Загружает дневные OHLCV с MOEX ISS для одной бумаги."""
    data = apimoex.get_board_history(
        session, ticker,
        start=start, end=end,
        board=board, market=market,
        columns=("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"),
    )
    if not data:
        logger.warning("MOEX: no data for %s", ticker)
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "TRADEDATE": "date",
        "OPEN": "open", "HIGH": "high", "LOW": "low",
        "CLOSE": "close", "VOLUME": "volume",
    })
    df["ticker"]   = ticker
    df["market"]   = "MOEX"
    df["currency"] = "RUB"
    df["date"]     = df["date"].astype(str)
    return df.dropna(subset=["close"])


def _fetch_imoex(session: requests.Session, start: str, end: str) -> pd.DataFrame:
    """IMOEX торгуется на другом борде (SNDX / индексы)."""
    data = apimoex.get_board_history(
        session, "IMOEX",
        start=start, end=end,
        board="SNDX", market="index",
        columns=("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"),
    )
    if not data:
        logger.warning("MOEX: no data for IMOEX")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "TRADEDATE": "date",
        "OPEN": "open", "HIGH": "high", "LOW": "low",
        "CLOSE": "close", "VOLUME": "volume",
    })
    df["ticker"]   = "IMOEX"
    df["market"]   = "MOEX"
    df["currency"] = "RUB"
    df["date"]     = df["date"].astype(str)
    return df.dropna(subset=["close"])


def _fetch_usdrub(session: requests.Session, start: str, end: str) -> pd.DataFrame:
    """
    USD/RUB через MOEX ISS REST API (USD000UTSTOM, борд CETS, engine currency).
    apimoex.get_board_history не поддерживает engine=currency, поэтому используем
    прямой HTTP-запрос.
    """
    base_url = (
        "https://iss.moex.com/iss/history/engines/currency/markets/selt/"
        "boards/CETS/securities/USD000UTSTOM.json"
    )
    rows = []
    start_row = 0
    batch = 100

    while True:
        params = {
            "from": start, "till": end,
            "iss.meta": "off", "iss.only": "history",
            "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,WAPRICE",
            "start": start_row,
            "limit": batch,
        }
        resp = session.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("history", {}).get("data", [])
        cols = payload.get("history", {}).get("columns", [])
        if not data:
            break
        rows.extend(data)
        if len(data) < batch:
            break
        start_row += batch

    if not rows:
        logger.warning("MOEX: no data for USDRUB_TOM (USD000UTSTOM)")
        return pd.DataFrame()

    col_map = {c: i for i, c in enumerate(cols)}
    records = []
    for row in rows:
        records.append({
            "date":   row[col_map["TRADEDATE"]],
            "open":   row[col_map.get("OPEN", -1)] if col_map.get("OPEN") is not None else None,
            "high":   row[col_map.get("HIGH", -1)] if col_map.get("HIGH") is not None else None,
            "low":    row[col_map.get("LOW", -1)]  if col_map.get("LOW")  is not None else None,
            "close":  row[col_map.get("CLOSE", col_map.get("WAPRICE", -1))],
            "volume": None,
        })

    df = pd.DataFrame(records)
    df["ticker"]   = "USDRUB"
    df["market"]   = "MOEX"
    df["currency"] = "RUB"
    df["date"]     = df["date"].astype(str)
    return df.dropna(subset=["close"])


def fetch_moex_all(start: str, end: str) -> pd.DataFrame:
    """Загружает SBER, LKOH, GAZP, IMOEX, USDRUB за период."""
    equity_tickers = ["SBER", "LKOH", "GAZP"]
    with requests.Session() as session:
        frames = []
        for t in equity_tickers:
            logger.info("Fetching MOEX: %s", t)
            frames.append(_fetch_moex_security(session, t, start, end))
        logger.info("Fetching MOEX: IMOEX")
        frames.append(_fetch_imoex(session, start, end))
        logger.info("Fetching MOEX: USDRUB_TOM")
        frames.append(_fetch_usdrub(session, start, end))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)



def fetch_us_all(start: str, end: str) -> pd.DataFrame:
    """Загружает NVDA, MSFT, AAPL, SPY через yfinance."""
    logger.info("Fetching yfinance: %s", US_TICKERS)
    raw = yf.download(
        US_TICKERS,
        start=start, end=end,
        auto_adjust=True,
        progress=False,
    )
    frames = []
    for ticker in US_TICKERS:
        try:
            ohlcv = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
        except KeyError:
            # single-ticker download has flat columns
            ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]]
        ohlcv = ohlcv.rename(columns=str.lower)
        ohlcv = ohlcv.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
        ohlcv["date"]     = ohlcv["date"].astype(str).str[:10]
        ohlcv["ticker"]   = ticker
        ohlcv["market"]   = "US"
        ohlcv["currency"] = "USD"
        frames.append(ohlcv.dropna(subset=["close"]))
    return pd.concat(frames, ignore_index=True)



def run_fetch(start: str = DEFAULT_START, end: str | None = None) -> None:
    if end is None:
        end = str(date.today())
    init_db()
    logger.info("Fetching prices %s → %s", start, end)
    moex_df = fetch_moex_all(start, end)
    us_df   = fetch_us_all(start, end)
    combined = pd.concat([moex_df, us_df], ignore_index=True)
    with get_db() as conn:
        upsert_prices(combined, conn)
    logger.info("Saved %d rows", len(combined))


def run_update() -> None:
    """Обновляет данные начиная с последней доступной даты в БД."""
    from src.db.repository import load_prices
    all_tickers = ["SBER", "LKOH", "GAZP", "IMOEX", "USDRUB", "NVDA", "MSFT", "AAPL", "SPY"]
    with get_db() as conn:
        df = load_prices(all_tickers, "2000-01-01", str(date.today()), conn)
    if df.empty:
        run_fetch()
        return
    last_date = str((pd.to_datetime(df["date"].max()) + timedelta(days=1)).date())
    run_fetch(start=last_date)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Только свежие данные")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end",   default=None)
    args = parser.parse_args()
    if args.update:
        run_update()
    else:
        run_fetch(start=args.start, end=args.end)
