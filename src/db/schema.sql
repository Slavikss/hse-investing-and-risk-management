-- RiskPulse SQLite schema

CREATE TABLE IF NOT EXISTS price_ohlcv (
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,   -- ISO-8601 YYYY-MM-DD
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL    NOT NULL,
    volume      REAL,
    market      TEXT    NOT NULL,   -- 'MOEX' | 'US'
    currency    TEXT    NOT NULL,   -- 'RUB' | 'USD'
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS macro_series (
    series_id   TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    value       REAL    NOT NULL,
    source      TEXT,               -- 'FRED' | 'MOEX'
    PRIMARY KEY (series_id, date)
);

CREATE TABLE IF NOT EXISTS raw_news (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash    TEXT    UNIQUE NOT NULL,
    source_domain TEXT,
    title       TEXT,
    text        TEXT,
    published_at TEXT
);

CREATE INDEX IF NOT EXISTS ix_price_date   ON price_ohlcv (date);
CREATE INDEX IF NOT EXISTS ix_macro_series ON macro_series (series_id, date);
