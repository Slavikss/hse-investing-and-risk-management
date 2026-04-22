"""CLI: загрузка/обновление данных, расчёты, бэктест, Streamlit (--help)."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("riskpulse")


PORTFOLIO_WEIGHTS = {
    "SBER": 0.15,
    "LKOH": 0.15,
    "GAZP": 0.10,
    "NVDA": 0.20,
    "MSFT": 0.20,
    "AAPL": 0.20,
}

DATE_START = "2019-01-01"



def step_fetch_prices() -> None:
    logger.info("=== STEP 1: Fetch prices (MOEX + yfinance) ===")
    from src.ingestion.price_fetcher import run_fetch
    run_fetch(start=DATE_START)


def step_fetch_macro(fred_key: str) -> None:
    logger.info("=== STEP 2: Fetch macro (FRED + CBR) ===")
    from src.ingestion.macro_fetcher import run_fetch
    run_fetch(fred_key=fred_key, start=DATE_START)


def step_compute(print_results: bool = True) -> None:
    """Вычисляет VaR, Attribution и печатает сводку в консоль."""
    import numpy as np
    import pandas as pd

    logger.info("=== STEP 3: Compute risk metrics ===")
    from src.db.repository import get_db, load_close_pivot, load_macro
    from src.risk.var import compute_all_var
    from src.risk.attribution import full_attribution

    end = str(__import__("datetime").date.today())
    with get_db() as conn:
        prices_raw = load_close_pivot(
            ["SBER", "LKOH", "GAZP", "NVDA", "MSFT", "AAPL", "IMOEX", "SPY", "USDRUB"],
            DATE_START, end, conn,
        )

    if prices_raw.empty:
        logger.error("No price data in DB. Run --fetch first.")
        return

    # Логарифмические доходности
    import numpy as _np
    rets_all = _np.log(prices_raw / prices_raw.shift(1)).dropna(how="all")

    # FX конвертация
    usdrub_ret = pd.Series(dtype=float)
    if "USDRUB" in rets_all.columns:
        usdrub_ret = rets_all["USDRUB"]
        rets_usd = rets_all.copy()
        for t in ["SBER", "LKOH", "GAZP"]:
            if t in rets_usd.columns:
                rets_usd[t] = rets_usd[t] - usdrub_ret
        rets_usd = rets_usd.drop(columns=["USDRUB", "IMOEX", "SPY"], errors="ignore")
    else:
        rets_usd = rets_all.drop(columns=["IMOEX", "SPY"], errors="ignore")

    tickers_in = [t for t in PORTFOLIO_WEIGHTS if t in rets_usd.columns]
    if not tickers_in:
        logger.error("None of portfolio tickers found in data")
        return

    logger.info("Computing VaR (1-day, 95%%)...")
    var1 = compute_all_var(rets_usd, PORTFOLIO_WEIGHTS, conf=0.95, horizon=1, n_sim=5000)

    logger.info("Computing VaR (10-day, 95%%)...")
    var10 = compute_all_var(rets_usd, PORTFOLIO_WEIGHTS, conf=0.95, horizon=10, n_sim=5000)

    logger.info("Computing risk attribution...")
    attr, limit_check = full_attribution(rets_usd, PORTFOLIO_WEIGHTS, usdrub_ret, conf=0.95, var_limit=0.02)

    if print_results:
        print("\n" + "=" * 60)
        print("RISKPULSE — RISK METRICS SUMMARY")
        print("=" * 60)
        print(f"\nPortfolio: {tickers_in}")
        print(f"Data range: {rets_usd.index.min().date()} → {rets_usd.index.max().date()}")
        print(f"Observations: {len(rets_usd)}")

        print("\n── VaR & ES (1-day, 95%) ──────────────────────────────────")
        s = var1.to_series()
        for k, v in s.items():
            if "nu" in str(k).lower() or _np.isnan(v):
                continue
            print(f"  {k:<25}: {v:.4%}")
        print(f"  Student-t ν (MLE)        : {var1.nu_t:.2f}")

        print("\n── VaR (10-day, 95%) overlapping ──────────────────────────")
        print(f"  Historical VaR 10d       : {var10.hist_var:.4%}")
        print(f"  Parametric VaR 10d       : {var10.param_var:.4%}")
        print(f"  MC Student-t VaR 10d     : {var10.mc_t_var:.4%}")
        if not _np.isnan(var10.sqrt10_param_var):
            print(f"  Basel √10 VaR (compare)  : {var10.sqrt10_param_var:.4%}")
            print(f"  Расхождение              : {abs(var10.param_var - var10.sqrt10_param_var):.4%}")

        print("\n── Risk Attribution (Component VaR) ────────────────────────")
        attr_df = attr.to_dataframe()
        print(attr_df.to_string(index=False))

        print("\n── VaR-лимит (2%) ──────────────────────────────────────────")
        if limit_check["breach"]:
            print(f"  ⚠️  НАРУШЕНИЕ: Portfolio VaR {limit_check['portfolio_var']:.4%} > {limit_check['limit']:.4%}")
            swr = limit_check.get("suggested_weight_reduction")
            if swr:
                print(f"  Top contributor: {swr['ticker']}")
                print(f"  Рекомендация: снизить вес {swr['current_w']:.2%} → {swr['suggested_w']:.2%}")
        else:
            print(f"  ✅ В рамках лимита: {limit_check['portfolio_var']:.4%} ≤ {limit_check['limit']:.4%}")
        print("=" * 60)


def step_backtest(conf: float = 0.95) -> None:
    """Запускает бэктест и выводит результаты Купича."""
    import numpy as np
    import pandas as pd

    logger.info("=== STEP 4: Backtest + Kupiec ===")
    from src.db.repository import get_db, load_close_pivot
    from src.backtest.runner import run_backtest

    end = str(__import__("datetime").date.today())
    with get_db() as conn:
        prices_raw = load_close_pivot(
            ["SBER", "LKOH", "GAZP", "NVDA", "MSFT", "AAPL", "USDRUB"],
            "2019-01-01", end, conn,
        )

    rets_all = np.log(prices_raw / prices_raw.shift(1)).dropna(how="all")
    if "USDRUB" in rets_all.columns:
        usdrub_ret = rets_all["USDRUB"]
        for t in ["SBER", "LKOH", "GAZP"]:
            if t in rets_all.columns:
                rets_all[t] = rets_all[t] - usdrub_ret
        rets_usd = rets_all.drop(columns=["USDRUB"], errors="ignore")
    else:
        rets_usd = rets_all

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS — COVID-2020")
    print("=" * 60)

    bt = run_backtest(
        rets_usd, PORTFOLIO_WEIGHTS, period="COVID_2020",
        conf=conf, methods=["historical", "parametric", "mc_normal", "mc_t"],
        n_sim=1000,
    )
    print(bt.kupiec_table.to_string(index=False))

    for r in bt.kupiec_results:
        print(f"\n{r.summary()}")


def step_dashboard() -> None:
    """Запускает Streamlit дашборд."""
    import subprocess
    app_path = str(_ROOT / "src" / "app" / "streamlit_app.py")
    logger.info("Launching Streamlit dashboard: %s", app_path)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "false"],
        check=False,
    )



def main() -> None:
    parser = argparse.ArgumentParser(
        description="RiskPulse — quantitative risk engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fetch",     action="store_true", help="Загрузить цены и макро")
    parser.add_argument("--update",    action="store_true", help="Обновить цены (инкремент)")
    parser.add_argument("--compute",   action="store_true", help="Вычислить риск-метрики")
    parser.add_argument("--backtest",  action="store_true", help="Запустить бэктест + Kupiec")
    parser.add_argument("--dashboard", action="store_true", help="Запустить Streamlit")
    parser.add_argument("--all",       action="store_true", help="Всё: fetch + compute + backtest")
    parser.add_argument("--fred-key",  default=os.environ.get("FRED_API_KEY", ""),
                        help="FRED API key (или переменная FRED_API_KEY)")
    parser.add_argument("--conf",      type=float, default=0.95)
    args = parser.parse_args()

    if not any([args.fetch, args.update, args.compute, args.backtest, args.dashboard, args.all]):
        parser.print_help()
        return

    if args.all or args.fetch:
        step_fetch_prices()
        if args.fred_key:
            step_fetch_macro(args.fred_key)
        else:
            logger.warning("FRED key not provided — macro data skipped")

    if args.update:
        from src.ingestion.price_fetcher import run_update
        run_update()

    if args.all or args.compute:
        step_compute()

    if args.all or args.backtest:
        step_backtest(conf=args.conf)

    if args.dashboard:
        step_dashboard()


if __name__ == "__main__":
    main()
