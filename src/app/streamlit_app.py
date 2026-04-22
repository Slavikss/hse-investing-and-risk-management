"""RiskPulse: Streamlit-дашборд. Запуск: streamlit run src/app/streamlit_app.py"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.db.repository import get_db, init_db, load_close_pivot, load_macro

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS = {
    "SBER": 0.15,
    "LKOH": 0.15,
    "GAZP": 0.10,
    "NVDA": 0.20,
    "MSFT": 0.20,
    "AAPL": 0.20,
}

ALL_TICKERS  = list(DEFAULT_WEIGHTS.keys())
RU_TICKERS   = ["SBER", "LKOH", "GAZP"]
US_TICKERS   = ["NVDA", "MSFT", "AAPL"]
DATE_START   = "2019-01-01"

st.set_page_config(
    page_title="RiskPulse — Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 8px;
    padding: 16px;
    border: 1px solid #313244;
    margin-bottom: 8px;
}
.metric-label { font-size: 12px; color: #a6adc8; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: bold; color: #cdd6f4; }
.metric-sub   { font-size: 12px; color: #6c7086; }
.breach-alert { background: #45273a; border: 1px solid #f38ba8;
                border-radius: 6px; padding: 12px; color: #f38ba8; }
.pass-alert   { background: #1e3a2f; border: 1px solid #a6e3a1;
                border-radius: 6px; padding: 12px; color: #a6e3a1; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_all_data(start: str = DATE_START) -> dict:
    end = str(pd.Timestamp.today().date())
    init_db()

    with get_db() as conn:
        prices_raw = load_close_pivot(
            ALL_TICKERS + ["IMOEX", "SPY", "USDRUB"],
            start, end, conn,
        )
        try:
            fedfunds = load_macro("FEDFUNDS", start, end, conn)
        except Exception:
            fedfunds = pd.Series(dtype=float)
        try:
            cbrate = load_macro("CBRATE", start, end, conn)
        except Exception:
            cbrate = pd.Series(dtype=float)
        try:
            brent = load_macro("DCOILBRENTEU", start, end, conn)
        except Exception:
            brent = pd.Series(dtype=float)

    return {
        "prices": prices_raw,
        "fedfunds": fedfunds,
        "cbrate": cbrate,
        "brent": brent,
    }


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def build_usd_returns(
    returns: pd.DataFrame,
    usdrub_col: str = "USDRUB",
) -> pd.DataFrame:
    if usdrub_col not in returns.columns:
        return returns.copy()

    fx_ret = returns[usdrub_col]
    out = returns.copy()
    for t in RU_TICKERS:
        if t in out.columns:
            out[t] = out[t] - fx_ret
    return out.drop(columns=[usdrub_col], errors="ignore")



def render_sidebar() -> tuple[dict[str, float], float, float, int]:
    st.sidebar.header("⚙️ Настройки портфеля")

    st.sidebar.subheader("Веса (должны давать 100%)")
    for ticker in ALL_TICKERS:
        k = f"w_{ticker}"
        if k not in st.session_state:
            st.session_state[k] = float(DEFAULT_WEIGHTS.get(ticker, 1.0 / max(len(ALL_TICKERS), 1)))

    weights = {}
    total = 0.0
    for ticker in ALL_TICKERS:
        k = f"w_{ticker}"
        w = st.sidebar.slider(f"{ticker}", 0.0, 1.0, step=0.01, key=k)
        weights[ticker] = w
        total += w

    if total <= 0:
        weights = {t: float(DEFAULT_WEIGHTS[t]) for t in ALL_TICKERS}
        tw = sum(weights.values())
        weights = {t: v / tw for t, v in weights.items()}
        st.sidebar.caption("Сумма весов 0 — использованы веса по умолчанию.")
    else:
        weights = {t: v / total for t, v in weights.items()}
        st.sidebar.caption(f"Сырой ввод: {total:.0%} → после нормировки 100%")

    st.sidebar.subheader("Параметры риска")
    conf = st.sidebar.selectbox("Уровень доверия VaR", [0.95, 0.99], index=0)
    var_limit = st.sidebar.number_input("VaR-лимит (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    n_sim = st.sidebar.selectbox("Симуляции MC", [1000, 5000, 10000], index=0)

    st.sidebar.subheader("Данные")
    if st.sidebar.button("🔄 Обновить данные"):
        st.cache_data.clear()
        for t in ALL_TICKERS:
            st.session_state.pop(f"w_{t}", None)
        st.rerun()

    return weights, float(conf), float(var_limit), int(n_sim)



def page_portfolio_risk(
    returns_usd: pd.DataFrame,
    weights:     dict[str, float],
    conf:        float,
    var_limit:   float,
    n_sim:       int,
) -> None:
    st.header("📊 Портфель & Риск")

    from src.risk.var import compute_all_var
    from src.risk.attribution import compute_component_var, add_fx_decomposition

    with st.spinner("Вычисляю VaR..."):
        var1 = compute_all_var(returns_usd, weights, conf=conf, horizon=1, n_sim=n_sim)

    with st.spinner("10-дневный VaR (overlapping)..."):
        var10 = compute_all_var(returns_usd, weights, conf=conf, horizon=10, n_sim=n_sim)

    col1, col2, col3, col4, col5 = st.columns(5)
    conf_lbl = f"{int(conf*100)}%"

    def _metric(col, label, val, sub=""):
        col.metric(label, f"{val:.2%}" if not np.isnan(val) else "—", delta=sub)

    _metric(col1, f"Historical VaR {conf_lbl}", var1.hist_var)
    _metric(col2, f"Parametric VaR {conf_lbl}", var1.param_var)
    _metric(col3, f"MC Normal VaR {conf_lbl}", var1.mc_norm_var)
    _metric(col4, f"MC Student-t VaR {conf_lbl}", var1.mc_t_var, f"ν={var1.nu_t:.1f}")
    _metric(col5, f"ES (CVaR) {conf_lbl}", var1.mc_t_es)

    st.divider()
    lim_check = var1.mc_t_var > var_limit
    if lim_check:
        st.markdown(
            f'<div class="breach-alert">⚠️ VaR-лимит НАРУШЕН: '
            f'Portfolio VaR {var1.mc_t_var:.2%} > лимит {var_limit:.2%}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="pass-alert">✅ VaR в рамках лимита: '
            f'{var1.mc_t_var:.2%} ≤ {var_limit:.2%}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Сравнение методов VaR")
        table = pd.DataFrame({
            "Метод": ["Historical", "Parametric EWMA", "MC Normal", "MC Student-t"],
            f"VaR 1d {conf_lbl}": [var1.hist_var, var1.param_var, var1.mc_norm_var, var1.mc_t_var],
            f"ES 1d {conf_lbl}":  [var1.hist_es,  var1.param_es,  var1.mc_norm_es,  var1.mc_t_es],
            f"VaR 10d {conf_lbl}": [var10.hist_var, var10.param_var, var10.mc_norm_var, var10.mc_t_var],
        })
        for col_name in table.columns[1:]:
            table[col_name] = table[col_name].apply(lambda x: f"{x:.3%}" if not np.isnan(x) else "—")
        st.dataframe(table, use_container_width=True, hide_index=True)

        if not np.isnan(var10.sqrt10_param_var):
            st.caption(
                f"Basel √10 scaling (comparison): {var10.sqrt10_param_var:.3%} "
                f"vs overlapping {var10.param_var:.3%} — "
                f"расхождение: {abs(var10.param_var - var10.sqrt10_param_var):.3%}"
            )

    with col_r:
        st.subheader("Компонентный VaR")
        try:
            attr = compute_component_var(returns_usd, weights, conf)
            pie_df = pd.DataFrame({
                "Актив": attr.tickers,
                "RC (%)": np.abs(attr.component_pct) * 100,
            })
            fig = px.pie(
                pie_df, values="RC (%)", names="Актив",
                title="Risk Contribution по активам",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось вычислить атрибуцию: {e}")

    st.subheader("FX-декомпозиция риска (RU кластер)")
    try:
        from src.risk.attribution import add_fx_decomposition
        usdrub_col_name = "USDRUB"
        data_raw = load_all_data()
        all_rets = compute_returns(data_raw["prices"])
        usdrub_returns = -all_rets.get(usdrub_col_name, pd.Series(dtype=float))
        attr2 = add_fx_decomposition(attr, returns_usd, usdrub_returns)
        attr_df = attr2.to_dataframe()
        attr_df["weight"] = attr_df["weight"].apply(lambda x: f"{x:.1%}")
        for c in ["component_var", "component_pct", "fx_component", "equity_component"]:
            if c in attr_df.columns:
                attr_df[c] = attr_df[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(attr_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"FX-декомпозиция недоступна: {e}")



def page_factor_analysis(
    prices:      pd.DataFrame,
    returns_usd: pd.DataFrame,
    data:        dict,
) -> None:
    st.header("📈 Факторный анализ")

    from src.risk.capm import (
        rolling_capm_ru,
        rolling_capm_global,
        rolling_ff3_global,
        load_french_factors,
    )

    tab_ru, tab_global, tab_ff3 = st.tabs(["RU кластер (CAPM)", "Global кластер (CAPM)", "Global FF3"])

    with tab_ru:
        st.subheader("CAPM: SBER / LKOH / GAZP vs IMOEX (RUB)")
        try:
            raw_prices = data["prices"]
            cbrate = data["cbrate"]
            prices_rub = raw_prices[["SBER", "LKOH", "GAZP"]].dropna(how="all")
            imoex     = raw_prices["IMOEX"].dropna()

            if cbrate.empty:
                cbrate = pd.Series(16.0, index=prices_rub.index)

            capm_ru = rolling_capm_ru(prices_rub, imoex, cbrate)

            if not capm_ru.empty:
                ticker_sel = st.selectbox("Тикер", ["SBER", "LKOH", "GAZP"], key="ru_ticker")
                df_t = capm_ru[capm_ru["ticker"] == ticker_sel].set_index("date")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_t.index, y=df_t["beta"], name="Beta (rolling 252d)", line=dict(color="#89b4fa")))
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="β=1")
                fig.update_layout(title=f"Rolling Beta: {ticker_sel} vs IMOEX", height=350,
                                  xaxis_title="Дата", yaxis_title="Beta")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_t.index, y=df_t["alpha"] * 252, name="Alpha (annualised)", line=dict(color="#a6e3a1")))
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                fig2.update_layout(title=f"Rolling Alpha (annualised): {ticker_sel}", height=300,
                                   xaxis_title="Дата", yaxis_title="Alpha")
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Актуальные коэффициенты CAPM (последняя доступная дата)")
                latest = capm_ru.sort_values("date").groupby("ticker").last().reset_index()
                latest["alpha_ann"] = latest["alpha"] * 252
                st.dataframe(
                    latest[["ticker", "alpha_ann", "beta", "r_squared"]].rename(columns={
                        "ticker": "Тикер", "alpha_ann": "Alpha (год.)", "beta": "Beta", "r_squared": "R²"
                    }).round(4),
                    use_container_width=True, hide_index=True,
                )
        except Exception as e:
            st.error(f"Ошибка RU CAPM: {e}")

    with tab_global:
        st.subheader("CAPM: NVDA / MSFT / AAPL vs SPY (USD)")
        try:
            raw_prices = data["prices"]
            fedfunds = data["fedfunds"]
            prices_usd_raw = raw_prices[["NVDA", "MSFT", "AAPL"]].dropna(how="all")
            spy = raw_prices["SPY"].dropna()

            if fedfunds.empty:
                fedfunds = pd.Series(5.0, index=prices_usd_raw.index)

            capm_gl = rolling_capm_global(prices_usd_raw, spy, fedfunds)

            if not capm_gl.empty:
                ticker_sel2 = st.selectbox("Тикер", ["NVDA", "MSFT", "AAPL"], key="gl_ticker")
                df_t2 = capm_gl[capm_gl["ticker"] == ticker_sel2].set_index("date")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_t2.index, y=df_t2["beta"], name="Beta (rolling 252d)", line=dict(color="#f9e2af")))
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig.update_layout(title=f"Rolling Beta: {ticker_sel2} vs SPY", height=350)
                st.plotly_chart(fig, use_container_width=True)

                latest_g = capm_gl.sort_values("date").groupby("ticker").last().reset_index()
                latest_g["alpha_ann"] = latest_g["alpha"] * 252
                st.dataframe(
                    latest_g[["ticker", "alpha_ann", "beta", "r_squared"]].rename(columns={
                        "ticker": "Тикер", "alpha_ann": "Alpha (год.)", "beta": "Beta", "r_squared": "R²"
                    }).round(4),
                    use_container_width=True, hide_index=True,
                )
        except Exception as e:
            st.error(f"Ошибка Global CAPM: {e}")

    with tab_ff3:
        st.subheader("Fama-French 3-Factor: NVDA / MSFT / AAPL (USD)")
        try:
            raw_prices = data["prices"]
            prices_usd_raw = raw_prices[["NVDA", "MSFT", "AAPL"]].dropna(how="all")

            with st.spinner("Загружаю факторы Ken French..."):
                ff3 = load_french_factors()

            ff3_result = rolling_ff3_global(prices_usd_raw, ff3)

            if not ff3_result.empty:
                ticker_ff3 = st.selectbox("Тикер", ["NVDA", "MSFT", "AAPL"], key="ff3_ticker")
                df_ff3 = ff3_result[ff3_result["ticker"] == ticker_ff3].set_index("date")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ff3.index, y=df_ff3["beta"], name="β_mkt", line=dict(color="#89b4fa")))
                fig.add_trace(go.Scatter(x=df_ff3.index, y=df_ff3["beta_smb"], name="β_SMB", line=dict(color="#a6e3a1")))
                fig.add_trace(go.Scatter(x=df_ff3.index, y=df_ff3["beta_hml"], name="β_HML", line=dict(color="#f38ba8")))
                fig.update_layout(title=f"FF3 Rolling Betas: {ticker_ff3}", height=400)
                st.plotly_chart(fig, use_container_width=True)

                latest_ff3 = ff3_result.sort_values("date").groupby("ticker").last().reset_index()
                latest_ff3["alpha_ann"] = latest_ff3["alpha"] * 252
                st.dataframe(
                    latest_ff3[["ticker", "alpha_ann", "beta", "beta_smb", "beta_hml", "r_squared"]].round(4),
                    use_container_width=True, hide_index=True,
                )
        except Exception as e:
            st.error(f"FF3 недоступен: {e}")



def page_stress_tests(
    returns_usd: pd.DataFrame,
    weights:     dict[str, float],
    data:        dict,
) -> None:
    st.header("🌩 Стресс-тесты")

    from src.risk.stress import (
        run_all_historical_scenarios,
        compute_covariance_matrices,
        build_hypothetical_shocks,
        apply_hypothetical_shock,
        stress_summary_table,
        avg_correlation,
        corr_from_cov,
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Исторические сценарии")
        with st.spinner("Вычисляю..."):
            scenarios = run_all_historical_scenarios(returns_usd, weights)

        for sc in scenarios:
            color = "#f38ba8" if sc.pnl_pct < 0 else "#a6e3a1"
            st.markdown(
                f"**{sc.name}** ({sc.period}): "
                f"<span style='color:{color};font-weight:bold'>{sc.pnl_pct:.2f}%</span>",
                unsafe_allow_html=True,
            )
            if sc.contrib:
                contrib_df = pd.DataFrame(
                    {"Актив": k, "Вклад (%)": round(v * 100, 3)}
                    for k, v in sc.contrib.items()
                )
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    with col_r:
        st.subheader("Normal vs Crisis корреляции")
        try:
            cov_mats = compute_covariance_matrices(returns_usd)
            tickers  = list(returns_usd.columns)

            regime = st.radio("Режим", ["normal", "crisis", "full"], horizontal=True)
            cov    = cov_mats[regime]
            corr   = corr_from_cov(cov)
            corr_df = pd.DataFrame(corr, index=tickers, columns=tickers)

            fig = px.imshow(
                corr_df.round(2),
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"Корреляционная матрица — {regime}",
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            avg_n = avg_correlation(cov_mats["normal"])
            avg_c = avg_correlation(cov_mats["crisis"])

            def _fmt_r(x: float) -> str:
                return f"{x:.3f}" if np.isfinite(x) else "—"

            st.metric("Средняя корреляция (normal)", _fmt_r(avg_n))
            dlt = None
            if np.isfinite(avg_n) and np.isfinite(avg_c):
                dlt = f"{avg_c - avg_n:+.3f} к кризису"
            st.metric("Средняя корреляция (crisis)", _fmt_r(avg_c), delta=dlt)
        except Exception as e:
            st.error(f"Ошибка ковариаций: {e}")

    st.subheader("Гипотетические шоки")
    try:
        raw_prices = data["prices"]
        all_rets   = compute_returns(raw_prices)
        usdrub_ret = -all_rets.get("USDRUB", pd.Series(dtype=float))
        brent_ret  = data["brent"].pct_change().dropna() if not data["brent"].empty else None

        shocks = build_hypothetical_shocks(returns_usd, usdrub_ret, brent_ret)

        shock_rows = []
        for shock in shocks:
            res = apply_hypothetical_shock(shock, weights)
            shock_rows.append({
                "Шок": shock.name,
                "Описание": shock.description,
                "Δ Фактор": f"{shock.delta:+.0%}",
                "P&L портфеля (%)": f"{res.get('total', 0) * 100:.2f}%",
            })
        st.dataframe(pd.DataFrame(shock_rows), use_container_width=True, hide_index=True)

        fig = go.Figure(go.Bar(
            x=[r["Шок"] for r in shock_rows],
            y=[float(r["P&L портфеля (%)"].replace("%", "")) for r in shock_rows],
            marker_color=["#f38ba8" if float(r["P&L портфеля (%)"].replace("%", "")) < 0
                          else "#a6e3a1" for r in shock_rows],
        ))
        fig.update_layout(title="P&L портфеля по гипотетическим шокам (%)",
                          yaxis_title="P&L (%)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка гипотетических шоков: {e}")



def page_backtest(
    returns_usd: pd.DataFrame,
    weights:     dict[str, float],
    conf:        float,
) -> None:
    st.header("🔬 Бэктест VaR")

    st.info(
        "Expanding window: VaR по истории до даты t, P&L в t+1. "
        "Период COVID_2020; санкции 2022 в этом бэктесте не используются (пропуск торгов MOEX)."
    )

    period = st.selectbox(
        "Период бэктеста",
        ["COVID_2020", "RateHike_2022"],
        key="bt_period",
    )
    methods_sel = st.multiselect(
        "Методы VaR",
        ["historical", "parametric", "mc_normal", "mc_t"],
        default=["historical", "parametric"],
        key="bt_methods",
    )

    if not methods_sel:
        st.warning("Выберите хотя бы один метод")
        return

    with st.spinner("Запускаю бэктест (expanding window)..."):
        from src.backtest.runner import run_backtest
        try:
            bt = run_backtest(returns_usd, weights, period=period, conf=conf,
                              methods=methods_sel, n_sim=1000)
        except Exception as e:
            st.error(f"Ошибка бэктеста: {e}")
            return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt.portfolio_pnl.index, y=bt.portfolio_pnl.values * 100,
        name="P&L (%)", line=dict(color="#cdd6f4", width=1),
    ))
    colors = {"historical": "#89b4fa", "parametric": "#a6e3a1",
              "mc_normal": "#f9e2af", "mc_t": "#f38ba8"}
    for method, vs in bt.var_series.items():
        fig.add_trace(go.Scatter(
            x=vs.index, y=-vs.values * 100,
            name=f"−VaR {method}",
            line=dict(color=colors.get(method, "#6c7086"), dash="dash"),
        ))
    fig.update_layout(
        title=f"P&L vs VaR — {period}",
        xaxis_title="Дата", yaxis_title="P&L / VaR (%)", height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    exceed_df = pd.concat(bt.exceedances.values(), axis=1)
    exceed_df.columns = bt.exceedances.keys()
    exceed_sums = exceed_df.sum()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Число exceedances по методам")
        st.bar_chart(exceed_sums)

    with col2:
        st.subheader("Результаты тестов Купича и Кристофферсена")
        st.dataframe(bt.kupiec_table, use_container_width=True, hide_index=True)

    with st.expander("Детальные результаты тестов"):
        for r in bt.kupiec_results:
            st.code(r.summary())



def page_about() -> None:
    st.header("ℹ️ Методология RiskPulse")
    st.markdown(
        "Портфель в **USD**; RU-доходности: r_USD ≈ r_RUB − Δln(USDRUB). "
        "VaR: historical, EWMA-parametric, MC normal, MC Student-t; 10d — overlapping. "
        "CAPM RU vs IMOEX (RUB), global vs SPY; FF3 по Ken French. "
        "Бэктест: Kupiec (LR_uc), Christoffersen (LR_ind), совместный LR_cc. "
        "MOEX Feb–Mar 2022 в бэктесте VaR не используется; стресс и FX — линейные приближения."
    )



def main() -> None:
    st.title("📊 RiskPulse — Portfolio Risk Dashboard")
    st.caption("ВШЭ · инвестиции и риск-менеджмент")

    weights, conf, var_limit, n_sim = render_sidebar()

    with st.spinner("Загружаю данные из SQLite..."):
        try:
            data = load_all_data()
            prices = data["prices"]

            if prices.empty:
                st.error(
                    "База данных пуста. Запустите сначала:\n"
                    "```\npython run_pipeline.py --fetch\n```"
                )
                st.stop()

            all_rets = compute_returns(prices)
            returns_usd = build_usd_returns(all_rets, "USDRUB")
            returns_usd = returns_usd[[c for c in ALL_TICKERS if c in returns_usd.columns]]

        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")
            st.exception(e)
            st.stop()

    page = st.sidebar.radio(
        "Раздел",
        ["📊 Портфель & Риск", "📈 Факторный анализ", "🌩 Стресс-тесты", "🔬 Бэктест", "ℹ️ О системе"],
        index=0,
    )

    if page == "📊 Портфель & Риск":
        page_portfolio_risk(returns_usd, weights, conf, var_limit, n_sim)
    elif page == "📈 Факторный анализ":
        page_factor_analysis(prices, returns_usd, data)
    elif page == "🌩 Стресс-тесты":
        page_stress_tests(returns_usd, weights, data)
    elif page == "🔬 Бэктест":
        page_backtest(returns_usd, weights, conf)
    else:
        page_about()


if __name__ == "__main__":
    main()
