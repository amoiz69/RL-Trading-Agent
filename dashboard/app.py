"""
dashboard/app.py
----------------
Streamlit dashboard for the RL Trading Agent.
Supports any S&P 500 stock via dynamic data download + feature engineering.

How to run locally
    streamlit run dashboard/app.py

How to deploy
    1. Push repo to GitHub (public)
    2. Go to share.streamlit.io
    3. New app → select repo → main file: dashboard/app.py
    4. Deploy
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Path setup — works both locally and on Streamlit Cloud
# ------------------------------------------------------------------ #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN, PPO
from env.trading_env import TradingEnv
from backtest.backtester import run_backtest, compute_metrics, compute_bnh_metrics
from data.pipeline import fetch_and_process


# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title  = "RL Trading Agent",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ------------------------------------------------------------------ #
# S&P 500 ticker list — fetched from Wikipedia once per session
# ------------------------------------------------------------------ #

@st.cache_data(ttl=60 * 60 * 24)   # refresh once a day
def get_sp500_tickers() -> list[str]:
    """Fetch the current S&P 500 constituent list from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)
    except Exception:
        # Fallback: a curated subset of well-known S&P 500 names
        return sorted([
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA",
            "BRK-B", "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD",
            "CVX", "MRK", "ABBV", "LLY", "PEP", "KO", "COST", "AVGO",
            "WMT", "MCD", "BAC", "PFE", "CRM", "TMO", "CSCO", "ABT",
            "ACN", "NKE", "DHR", "DIS", "NEE", "VZ", "TXN", "ADBE",
            "HON", "RTX", "AMGN", "QCOM", "UPS", "IBM", "GS", "AXP",
            "SPGI", "INTC", "CAT", "MS", "BLK", "NOW", "INTU", "SBUX",
            "ELV", "MDT", "PLD", "LMT", "CI", "MO", "GILD", "ZTS",
            "T", "REGN", "AON", "ISRG", "DE", "SYK", "MDLZ", "MMC",
            "D", "EW", "CL", "HCA", "WM", "MRNA", "NSC", "USB",
        ])


# ------------------------------------------------------------------ #
# Helpers — cached so they only run once per session
# ------------------------------------------------------------------ #

@st.cache_resource
def load_model(agent: str, model_variant: str = "aapl"):
    """
    Load a trained SB3 model.
    model_variant: 'aapl'  → models/ppo_aapl.zip or dqn_aapl.zip
                   'multi' → models/ppo_multi.zip  (PPO only)
    """
    if model_variant == "multi":
        # Multi-stock model is PPO only
        path = os.path.join(PROJECT_ROOT, "models", "ppo_multi.zip")
    else:
        path = os.path.join(PROJECT_ROOT, "models", f"{agent.lower()}_aapl.zip")
    if not os.path.exists(path):
        return None
    cls = PPO  # multi is always PPO; aapl uses the right class
    if model_variant == "aapl":
        cls = DQN if agent == "DQN" else PPO
    return cls.load(path)


@st.cache_data(show_spinner=False)
def load_ticker_data(ticker: str):
    """
    Download + feature-engineer + scale data for any ticker.
    Returns (splits_dict, raw_splits_dict, raw_full_df).
    Cached per ticker per session.
    """
    train_df, val_df, test_df, raw_df = fetch_and_process(ticker)

    splits = {}
    raw_splits = {}

    if len(train_df) > 0:
        splits["train"]     = train_df
        raw_splits["train"] = raw_df.loc[train_df.index]
    if len(val_df) > 0:
        splits["val"]       = val_df
        raw_splits["val"]   = raw_df.loc[val_df.index]
    if len(test_df) > 0:
        splits["test"]      = test_df
        raw_splits["test"]  = raw_df.loc[test_df.index]

    return splits, raw_splits, raw_df


def _fmt_daterange(df: pd.DataFrame) -> str:
    """Return 'YYYY – YYYY' formatted date range from a DataFrame index."""
    return f"{df.index[0].year} – {df.index[-1].year}"


# ------------------------------------------------------------------ #
# Chart builders
# ------------------------------------------------------------------ #

def build_trade_chart(env: TradingEnv, raw_df: pd.DataFrame,
                      window_size: int, agent_name: str) -> go.Figure:
    """Plotly price chart with buy/sell markers and portfolio overlay."""
    prices     = raw_df["Close"].values
    dates      = raw_df.index
    trade_log  = env.get_trade_log()
    portfolio  = pd.Series(env.portfolio_history)
    bnh_series = 10_000 * (raw_df["Close"] / raw_df["Close"].iloc[0])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        row_heights=[0.6, 0.4],
        subplot_titles=[
            f"{agent_name} — price chart with trade signals",
            "Portfolio value vs Buy & Hold",
        ],
        vertical_spacing=0.12,
    )

    # Price line
    fig.add_trace(
        go.Scatter(x=list(range(len(prices))), y=prices,
                   name="Price", line=dict(color="#5B8DB8", width=1)),
        row=1, col=1
    )

    # Buy / Sell markers
    if len(trade_log) > 0:
        buys  = trade_log[trade_log["action"] == "BUY"]
        sells = trade_log[trade_log["action"] == "SELL"]

        bidx = (buys["step"].values - window_size).clip(0, len(prices) - 1)
        sidx = (sells["step"].values - window_size).clip(0, len(prices) - 1)

        fig.add_trace(
            go.Scatter(
                x=bidx, y=prices[bidx],
                mode="markers",
                marker=dict(symbol="triangle-up", color="limegreen", size=10),
                name=f"Buy ({len(buys)})",
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=sidx, y=prices[sidx],
                mode="markers",
                marker=dict(symbol="triangle-down", color="crimson", size=10),
                name=f"Sell ({len(sells)})",
            ), row=1, col=1
        )

    # Portfolio vs Buy & Hold
    x_range = list(range(len(portfolio)))
    fig.add_trace(
        go.Scatter(x=x_range, y=portfolio.values,
                   name=agent_name,
                   line=dict(color="steelblue", width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_range,
                   y=bnh_series.values[:len(portfolio)],
                   name="Buy & Hold",
                   line=dict(color="gray", width=1, dash="dash")),
        row=2, col=1
    )
    fig.add_hline(y=10_000, line_dash="dot", line_color="black",
                  line_width=0.7, opacity=0.4, row=2, col=1)

    fig.update_layout(
        height=620,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

    return fig


def build_drawdown_chart(envs: dict, labels: dict) -> go.Figure:
    """Overlapping drawdown chart for multiple agents."""
    colors = {"DQN": "#4682b4", "PPO": "#ff8c00", "Buy & Hold": "#888888"}
    fig = go.Figure()

    for key, env_obj in envs.items():
        if env_obj is None:
            continue
        port = pd.Series(env_obj.portfolio_history)
        dd   = ((port - port.cummax()) / port.cummax()) * 100
        fig.add_trace(go.Scatter(
            x=list(range(len(dd))), y=dd.values,
            fill="tozeroy",
            name=key,
            line=dict(color=colors.get(key, "purple"), width=1),
            fillcolor=f"rgba({','.join(str(int(c*255)) for c in px.colors.hex_to_rgb(colors.get(key,'#888')[:7]))},0.15)",
        ))

    fig.update_layout(
        title="Drawdown comparison",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Drawdown (%)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def metric_delta_color(agent_val, bnh_val, lower_is_better=False):
    """Return delta string for st.metric."""
    if isinstance(agent_val, str) or isinstance(bnh_val, str):
        return None
    delta = agent_val - bnh_val
    if lower_is_better:
        delta = -delta
    return f"{delta:+.3f} vs B&H"


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #

with st.sidebar:
    st.title("RL Trading Agent")
    st.caption("DQN + PPO agents trained to trade stocks")
    st.divider()

    # --- Ticker selector ---
    sp500 = get_sp500_tickers()
    # Default to AAPL (index lookup, fallback to 0)
    default_idx = sp500.index("AAPL") if "AAPL" in sp500 else 0

    ticker = st.selectbox(
        "S&P 500 Ticker",
        options=sp500,
        index=default_idx,
        help="Select any S&P 500 stock. Data is downloaded automatically via yfinance.",
    )

    agent = st.selectbox("Agent", ["DQN", "PPO"])

    # -- Model variant selector ---
    multi_exists = os.path.exists(os.path.join(PROJECT_ROOT, "models", "ppo_multi.zip"))
    model_options = ["AAPL-only"]
    if multi_exists:
        model_options.append("Multi-stock (PPO)")
    else:
        model_options.append("Multi-stock (PPO) ← train first")

    model_label = st.selectbox(
        "Model",
        options=model_options,
        help=(
            "AAPL-only: original single-stock trained model.\n"
            "Multi-stock: generalised model trained on 8 diverse S&P 500 stocks.\n"
            "Train the multi-stock model: python agents/multi_stock_ppo.py --steps 100000"
        ),
        disabled=(not multi_exists and len(model_options) == 1),
    )
    model_variant   = "multi" if "Multi-stock" in model_label and multi_exists else "aapl"
    multi_selected  = (model_variant == "multi")

    # --- Load data first so we can build dynamic split labels ---
    data_ok = True
    try:
        splits, raw_splits, raw_full = load_ticker_data(ticker)
    except Exception as e:
        st.error(f"Failed to load data for **{ticker}**: {e}")
        data_ok = False

    # Build split options with real date ranges
    split_options = {}
    if data_ok:
        if "val" in splits and len(splits["val"]) > 0:
            split_options[f"Validation ({_fmt_daterange(splits['val'])})"] = "val"
        if "test" in splits and len(splits["test"]) > 0:
            split_options[f"Test ({_fmt_daterange(splits['test'])})"] = "test"

    split_label = st.selectbox(
        "Dataset split",
        options=list(split_options.keys()) if split_options else ["No splits available"],
    )
    split_key = split_options.get(split_label, "test")

    st.divider()
    run_button = st.button("▶ Run Agent", type="primary", use_container_width=True)

    st.divider()
    st.caption("**About this project**")
    st.caption(
        "End-to-end RL trading agent built with Stable-Baselines3, "
        "Gymnasium, and Streamlit. Supports any S&P 500 stock via "
        "live yfinance data."
    )
    st.caption("Phase 6 of a 6-phase ML portfolio project.")


# ------------------------------------------------------------------ #
# Banner — model info + transfer mode notice
# ------------------------------------------------------------------ #

if data_ok:
    if multi_selected:
        if agent == "DQN":
            st.warning(
                "⚠️ The **Multi-stock** model is PPO only. "
                "Switching Agent to **PPO** is recommended. "
                "(The multi-stock model will still run — it uses the PPO policy regardless.)"
            )
        if ticker != "AAPL":
            st.success(
                f"🌐 **Multi-stock model** — trained on AAPL, MSFT, TSLA, JPM, JNJ, XOM, AMZN & PG. "
                f"Applying to **{ticker}**. This model was designed to generalise across stocks."
            )
        else:
            st.success(
                "🌐 **Multi-stock model** — trained on 8 diverse S&P 500 stocks for better generalisation."
            )
    elif ticker != "AAPL":
        st.info(
            f"🔬 **Zero-shot transfer mode** — The AAPL-only models were trained exclusively on "
            f"**AAPL** data and are being applied to **{ticker}** without retraining. "
            f"Try the **Multi-stock** model (train it first) for better cross-stock performance.",
            icon="ℹ️",
        )


# ------------------------------------------------------------------ #
# Load model
# ------------------------------------------------------------------ #

model = load_model(agent, model_variant)

if model is None:
    if model_variant == "multi":
        st.error(
            "Multi-stock model not found: `models/ppo_multi.zip`\n\n"
            "Train it first by running:\n"
            "```\npython agents/multi_stock_ppo.py --steps 100000\n```"
        )
    else:
        st.error(
            f"Model not found: `models/{agent.lower()}_aapl.zip`\n\n"
            "Train the agent first by running Phase 4 (`phase4_train.ipynb`)."
        )
    st.stop()

if not data_ok:
    st.stop()

df     = splits.get(split_key)
raw_df = raw_splits.get(split_key)

if df is None or raw_df is None or len(df) == 0:
    st.error(f"No data available for **{ticker}** in the **{split_label}** period.")
    st.stop()

WINDOW_SIZE = 10


# ------------------------------------------------------------------ #
# Session state — persist results across reruns
# ------------------------------------------------------------------ #

if "results" not in st.session_state:
    st.session_state.results    = {}
    st.session_state.envs       = {}
    st.session_state.last_run   = None


# ------------------------------------------------------------------ #
# Run agent when button clicked
# ------------------------------------------------------------------ #

if run_button:
    run_key = f"{ticker}_{agent}_{split_key}_{model_variant}"

    with st.spinner(f"Running {agent} on {ticker} — {split_label}..."):
        env     = run_backtest(model, df, raw_df, WINDOW_SIZE)
        metrics = compute_metrics(env)
        bnh     = compute_bnh_metrics(raw_df)

        st.session_state.results[run_key] = (metrics, bnh)
        st.session_state.envs[run_key]    = env
        st.session_state.last_run         = run_key

    st.success(f"{agent} on {ticker} complete — {metrics['n_trades']} trades executed.")


# ------------------------------------------------------------------ #
# Main content — tabs
# ------------------------------------------------------------------ #

tab1, tab2, tab3 = st.tabs(["Agent run", "Metrics", "Tearsheet"])

run_key = st.session_state.get("last_run")

# Parse run_key helper
def _parse_run_key(rk: str | None):
    """Returns (ticker, agent, split, model_variant) or Nones."""
    if not rk:
        return None, None, None, None
    parts = rk.split("_")
    # Format: TICKER_AGENT_SPLIT_VARIANT  (variant may be absent for old keys)
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], parts[3]
    elif len(parts) == 3:
        return parts[0], parts[1], parts[2], "aapl"
    return None, None, None, None


# ================================================================== #
# TAB 1 — Agent run chart
# ================================================================== #
with tab1:
    if run_key and run_key in st.session_state.envs:
        env     = st.session_state.envs[run_key]
        metrics, bnh = st.session_state.results[run_key]

        # Quick headline metrics at the top
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final portfolio",
                  f"${metrics['final_value']:,.0f}",
                  delta=f"${metrics['final_value']-10_000:+,.0f}")
        c2.metric("Total return",
                  f"{metrics['total_return']:+.1f}%",
                  delta=metric_delta_color(metrics['total_return'], bnh['total_return']))
        c3.metric("Sharpe ratio",  f"{metrics['sharpe']:.3f}")
        c4.metric("Trades",        metrics['n_trades'])

        rk_ticker, rk_agent, rk_split, rk_variant = _parse_run_key(run_key)
        chart_label = f"{rk_agent} ({'Multi' if rk_variant == 'multi' else 'AAPL-only'})"
        st.plotly_chart(
            build_trade_chart(env, raw_df, WINDOW_SIZE, chart_label),
            use_container_width=True,
        )
        st.caption("▲ green = Buy   ▼ red = Sell")

    else:
        st.info("Configure settings in the sidebar and press **▶ Run Agent** to start.")


# ================================================================== #
# TAB 2 — Full metrics comparison
# ================================================================== #
with tab2:
    if run_key and run_key in st.session_state.results:
        metrics, bnh = st.session_state.results[run_key]
        env          = st.session_state.envs[run_key]

        rk_ticker, rk_agent, rk_split, rk_variant = _parse_run_key(run_key)
        rk_label   = "Validation" if rk_split == "val" else "Test"
        model_tag  = "Multi-stock" if rk_variant == "multi" else "AAPL-only"

        st.subheader(f"{rk_agent} ({model_tag}) vs Buy & Hold — {rk_ticker} {rk_label}")
        st.caption("Delta values show agent performance relative to Buy & Hold baseline.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total return (%)",
                      f"{metrics['total_return']:+.2f}%",
                      delta=f"{metrics['total_return'] - bnh['total_return']:+.2f}% vs B&H")
            st.metric("Win rate (%)",
                      f"{metrics['win_rate']:.1f}%")

        with c2:
            st.metric("Sharpe ratio",
                      f"{metrics['sharpe']:.3f}",
                      delta=f"{metrics['sharpe'] - bnh['sharpe']:+.3f} vs B&H")
            st.metric("Calmar ratio",
                      f"{metrics['calmar']:.3f}",
                      delta=f"{metrics['calmar'] - bnh['calmar']:+.3f} vs B&H")

        with c3:
            st.metric("Max drawdown (%)",
                      f"{metrics['max_drawdown']:.2f}%",
                      delta=f"{bnh['max_drawdown'] - metrics['max_drawdown']:+.2f}% vs B&H",
                      delta_color="inverse")
            st.metric("Trade frequency",
                      f"{metrics['trade_frequency']:.1f}%")

        st.divider()

        # If both DQN and PPO have been run for the same ticker+split+variant, show drawdown
        dqn_key = f"{rk_ticker}_DQN_{rk_split}_{rk_variant}"
        ppo_key = f"{rk_ticker}_PPO_{rk_split}_{rk_variant}"

        envs_for_dd = {}
        if dqn_key in st.session_state.envs:
            envs_for_dd["DQN"] = st.session_state.envs[dqn_key]
        if ppo_key in st.session_state.envs:
            envs_for_dd["PPO"] = st.session_state.envs[ppo_key]

        if len(envs_for_dd) > 0:
            class _BnHEnv:
                def __init__(self, raw):
                    prices = raw["Close"].reset_index(drop=True)
                    self.portfolio_history = (10_000 * prices / prices.iloc[0]).tolist()
            envs_for_dd["Buy & Hold"] = _BnHEnv(raw_df)
            st.plotly_chart(
                build_drawdown_chart(envs_for_dd, {}),
                use_container_width=True,
            )
            if len(envs_for_dd) < 3:
                st.caption("Run both DQN and PPO to see a full comparison in the drawdown chart.")

        # Full metrics table
        st.divider()
        st.subheader("Full metrics table")
        table_data = {
            "Metric"        : ["Final portfolio ($)", "Total return (%)", "Sharpe ratio",
                                "Max drawdown (%)",   "Calmar ratio",      "Win rate (%)",
                                "Trades executed",    "Trade frequency (%)"],
            f"{rk_agent} ({model_tag})" : [metrics["final_value"],   metrics["total_return"],
                                metrics["sharpe"],        metrics["max_drawdown"],
                                metrics["calmar"],        metrics["win_rate"],
                                metrics["n_trades"],      metrics["trade_frequency"]],
            "Buy & Hold"    : [bnh["final_value"],        bnh["total_return"],
                                bnh["sharpe"],             bnh["max_drawdown"],
                                bnh["calmar"],             bnh["win_rate"],
                                bnh["n_trades"],           bnh["trade_frequency"]],
        }
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Trade log
        st.divider()
        tlog = env.get_trade_log()
        if len(tlog) > 0:
            st.subheader(f"Trade log ({len(tlog)} trades)")
            st.dataframe(tlog, use_container_width=True, hide_index=True)
        else:
            st.info("No trades were executed in this episode.")

    else:
        st.info("Run the agent first (Tab 1 → sidebar → ▶ Run Agent).")


# ================================================================== #
# TAB 3 — QuantStats tearsheet
# ================================================================== #
with tab3:
    if run_key:
        rk_agent_tab3 = run_key.split("_")[1]   # DQN or PPO
        report_path   = os.path.join(
            PROJECT_ROOT, "reports",
            f"{rk_agent_tab3.lower()}_tearsheet.html"
        )

        if os.path.exists(report_path):
            st.subheader(f"{rk_agent_tab3} — QuantStats tearsheet")
            st.caption(
                "Full performance report generated by QuantStats. "
                "Includes rolling Sharpe, monthly returns heatmap, and worst drawdown periods."
            )
            with open(report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=900, scrolling=True)
        else:
            st.warning(
                f"Tearsheet not found at `reports/{rk_agent_tab3.lower()}_tearsheet.html`.\n\n"
                "Generate it by running **Phase 5** (`phase5_backtest.ipynb` → Cell 10)."
            )
    else:
        st.info("Run the agent first to view the tearsheet.")