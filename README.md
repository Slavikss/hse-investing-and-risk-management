# RiskPulse — Portfolio Risk Analytics System

ВШЭ · Инвестиции и риск-менеджмент · Track A: Quantitative Risk Engine

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Загрузить данные (MOEX + yfinance; FRED опционально)
python run_pipeline.py --fetch
# С макро-данными FRED:
python run_pipeline.py --fetch --fred-key YOUR_FRED_API_KEY

# 3. Вычислить риск-метрики
python run_pipeline.py --compute

# 4. Бэктест + Kupiec/Christoffersen
python run_pipeline.py --backtest

# 5. Дашборд
python run_pipeline.py --dashboard
# или напрямую:
streamlit run src/app/streamlit_app.py
```

## Структура проекта

```
riskpulse/
├── src/
│   ├── ingestion/
│   │   ├── price_fetcher.py   # MOEX ISS + yfinance + USDRUB_TOM
│   │   └── macro_fetcher.py   # FRED (FEDFUNDS, CPI, DXY, Brent) + CBR
│   ├── risk/
│   │   ├── capm.py            # Rolling CAPM (RU/RUB + Global/USD) + FF3
│   │   ├── var.py             # 4 метода VaR + ES + 10-day overlapping
│   │   ├── stress.py          # Normal/crisis covariance + сценарии
│   │   └── attribution.py     # Компонентный VaR + FX decomposition
│   ├── backtest/
│   │   ├── runner.py          # Expanding window backtest
│   │   └── kupiec.py          # Kupiec + Christoffersen тесты
│   ├── db/
│   │   ├── schema.sql
│   │   └── repository.py
│   └── app/
│       └── streamlit_app.py   # 5-экранный дашборд
├── data/                      # SQLite БД (создаётся автоматически)
├── run_pipeline.py            # CLI точка входа
└── requirements.txt
```

## Реальные числа (данные 2019-01-03 → 2026-04-21)

### VaR и ES (1-day, 95%, портфель SBER15/LKOH15/GAZP10/NVDA20/MSFT20/AAPL20)

| Метод            | VaR 1d  | ES 1d  | VaR 10d (overlapping) |
|------------------|---------|--------|----------------------|
| Historical       | 2.33%   | 3.70%  | 6.67%                |
| Parametric EWMA  | 1.63%   | 2.04%  | 7.19%                |
| MC Normal        | 1.55%   | 1.91%  | —                    |
| MC Student-t     | 1.66%   | 2.24%  | 5.76%                |

Student-t ν (MLE) = **9.16** → умеренные толстые хвосты

Basel √10 scaling: **5.14%** vs overlapping **7.19%** → расхождение **2.04%** (нарушение i.i.d. в стресс)

### Kupiec + Christoffersen (COVID-2020, T=317)

| Метод        | Exceedances | p_uc   | Reject_uc | p_ind  | Reject_ind | p_cc   |
|--------------|-------------|--------|-----------|--------|------------|--------|
| Historical   | 26 (8.20%)  | 0.0162 | ✗ ДА      | 0.0112 | ✗ ДА       | 0.0022 |
| Parametric   | 29 (9.15%)  | 0.0023 | ✗ ДА      | 0.3570 | ✓ нет      | 0.0062 |
| MC Normal    | 28 (8.83%)  | 0.0045 | ✗ ДА      | 0.2905 | ✓ нет      | 0.0102 |
| MC Student-t | 24 (7.57%)  | **0.0501** | ✓ **нет** | 0.3450 | ✓ нет  | **0.0940** |

**Вывод**: MC Student-t — единственная модель, прошедшая оба теста при α=5%.

### Risk Attribution (Component VaR, 95%)

| Тикер | Вес   | RC (%)  | FX компонента |
|-------|-------|---------|---------------|
| NVDA  | 20%   | 32.6%   | —             |
| AAPL  | 20%   | 18.3%   | —             |
| MSFT  | 20%   | 17.9%   | —             |
| SBER  | 15%   | 12.7%   | −0.001939     |
| LKOH  | 15%   | 10.7%   | −0.001662     |
| GAZP  | 10%   |  7.9%   | −0.001162     |

FX-компонента отрицательна для RU активов: ослабление рубля снижает USD-доходность,
но экспортёры (LKOH, GAZP) частично выигрывают через ценовой канал.

## Исследовательские вопросы

**RQ1** (валютный риск): При текущем составе портфеля (40% RU) FX компонента составляет
~31% суммарного риска RU кластера. Parametric VaR систематически недооценивает риск
в кризис (подтверждено Kupiec).

**RQ2** (геополитические шоки): Стресс-тест COVID-2020 показывает кумулятивный убыток
портфеля; sanctions-шок (Feb 2022) доступен только как гипотетический сценарий —
MOEX был закрыт 24 торговых дня.

## Ограничения

- MOEX закрыт 25 Feb – 24 Mar 2022 → sanctions 2022 только стресс-сценарий
- FF3 неприменим для RU (нет официальных факторов для MOEX)
- FX конвертация — линейное приближение: `r^USD ≈ r^RUB + r_FX`, ошибка < 0.1%/день
- Линейное P&L в стресс-тестах (без опционов, без convexity)
