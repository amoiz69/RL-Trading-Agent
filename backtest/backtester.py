"""
backtest/backtester.py
----------------------
Runs trained agents on the test set and computes all evaluation metrics.

How to run
    python backtest/backtester.py

What it produces
    reports/dqn_tearsheet.html   — full QuantStats HTML report
    reports/ppo_tearsheet.html   — full QuantStats HTML report
    Console comparison table     — DQN vs PPO vs Buy & Hold
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN, PPO
from env.trading_env import TradingEnv


# ------------------------------------------------------------------ #
# Core episode runner
# ------------------------------------------------------------------ #

def run_backtest(model, df: pd.DataFrame, raw_df: pd.DataFrame,
                 window_size: int = 10, initial_balance: float = 10_000.0) -> TradingEnv:
    """
    Run one full deterministic episode on any dataset split.

    Parameters
    ----------
    model     : trained SB3 model (DQN or PPO)
    df        : normalised feature DataFrame
    raw_df    : raw OHLCV DataFrame (real dollar prices)
    window_size : observation window
    initial_balance : starting portfolio value

    Returns
    -------
    env : completed TradingEnv with portfolio_history and trade_log populated
    """
    env = TradingEnv(df, raw_df,
                     initial_balance=initial_balance,
                     window_size=window_size)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    return env


# ------------------------------------------------------------------ #
# Metrics computation
# ------------------------------------------------------------------ #

def compute_metrics(env: TradingEnv,
                    initial_balance: float = 10_000.0) -> dict:
    """
    Compute all 6 evaluation metrics from a completed episode.

    Returns a dict with:
        final_value, total_return, sharpe, max_drawdown,
        calmar, win_rate, n_trades, trade_frequency
    """
    portfolio  = pd.Series(env.portfolio_history, dtype=float)
    returns    = portfolio.pct_change().dropna()
    trade_log  = env.get_trade_log()

    # -- Core metrics --------------------------------------------------
    final_value  = portfolio.iloc[-1]
    total_return = (final_value - initial_balance) / initial_balance * 100

    # Sharpe ratio (annualised, 252 trading days)
    sharpe = 0.0
    if returns.std() > 1e-8:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    # Max drawdown
    rolling_max  = portfolio.cummax()
    drawdown     = (portfolio - rolling_max) / (rolling_max + 1e-8)
    max_drawdown = drawdown.min() * 100

    # Calmar ratio (annualised return / |max drawdown|)
    n_years      = len(portfolio) / 252
    annual_ret   = ((final_value / initial_balance) ** (1 / max(n_years, 0.01)) - 1) * 100
    calmar        = annual_ret / abs(max_drawdown) if abs(max_drawdown) > 1e-4 else 0.0

    # Trade metrics
    n_trades        = len(trade_log)
    trade_frequency = n_trades / len(portfolio) * 100 if len(portfolio) > 0 else 0.0

    win_rate = 0.0
    if n_trades > 0 and "action" in trade_log.columns:
        buys  = trade_log[trade_log["action"] == "BUY"]
        sells = trade_log[trade_log["action"] == "SELL"]
        if len(buys) > 0 and len(sells) > 0:
            avg_buy_price = buys["price"].mean()
            wins          = (sells["price"] > avg_buy_price).sum()
            win_rate      = wins / len(sells) * 100

    return {
        "final_value"     : round(final_value, 2),
        "total_return"    : round(total_return, 2),
        "sharpe"          : round(float(sharpe), 3),
        "max_drawdown"    : round(float(max_drawdown), 2),
        "calmar"          : round(float(calmar), 3),
        "win_rate"        : round(win_rate, 1),
        "n_trades"        : n_trades,
        "trade_frequency" : round(trade_frequency, 1),
    }


def compute_bnh_metrics(raw_df: pd.DataFrame,
                        initial_balance: float = 10_000.0) -> dict:
    """
    Compute the same metrics for a simple Buy & Hold baseline.
    Invests full balance on day 1, holds until the end.
    """
    prices    = raw_df["Close"].reset_index(drop=True)
    portfolio = initial_balance * (prices / prices.iloc[0])
    returns   = portfolio.pct_change().dropna()

    final_value  = portfolio.iloc[-1]
    total_return = (final_value - initial_balance) / initial_balance * 100

    sharpe = 0.0
    if returns.std() > 1e-8:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    rolling_max  = portfolio.cummax()
    drawdown     = (portfolio - rolling_max) / (rolling_max + 1e-8)
    max_drawdown = drawdown.min() * 100

    n_years   = len(portfolio) / 252
    ann_ret   = ((final_value / initial_balance) ** (1 / max(n_years, 0.01)) - 1) * 100
    calmar    = ann_ret / abs(max_drawdown) if abs(max_drawdown) > 1e-4 else 0.0

    return {
        "final_value"     : round(final_value, 2),
        "total_return"    : round(total_return, 2),
        "sharpe"          : round(float(sharpe), 3),
        "max_drawdown"    : round(float(max_drawdown), 2),
        "calmar"          : round(float(calmar), 3),
        "win_rate"        : "N/A",
        "n_trades"        : 1,
        "trade_frequency" : "N/A",
    }


# ------------------------------------------------------------------ #
# QuantStats tearsheet
# ------------------------------------------------------------------ #

def generate_tearsheet(env: TradingEnv, raw_df: pd.DataFrame,
                        title: str, output_path: str):
    """
    Generate a full HTML tearsheet using QuantStats.
    Compares agent returns against SPY (S&P 500 benchmark).
    """
    try:
        import quantstats as qs

        portfolio = pd.Series(
            env.portfolio_history,
            index = raw_df.index[env.window_size : env.window_size + len(env.portfolio_history)],
            dtype = float,
        )
        returns = portfolio.pct_change().dropna()
        returns.index = pd.to_datetime(returns.index)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        qs.reports.html(
            returns,
            benchmark  = "SPY",
            output     = output_path,
            title      = title,
            download_filename = os.path.basename(output_path),
        )
        print(f"  Tearsheet saved → {output_path}")

    except ImportError:
        print("  [WARN] quantstats not installed. Skipping tearsheet.")
        print("         Install with: pip install quantstats")
    except Exception as e:
        print(f"  [WARN] Tearsheet generation failed: {e}")
        print("         This is often a yfinance/SPY data issue. Metrics are still valid.")


# ------------------------------------------------------------------ #
# Comparison table printer
# ------------------------------------------------------------------ #

def print_comparison(results: dict):
    metrics = [
        ("Final portfolio ($)", "final_value"),
        ("Total return (%)",    "total_return"),
        ("Sharpe ratio",        "sharpe"),
        ("Max drawdown (%)",    "max_drawdown"),
        ("Calmar ratio",        "calmar"),
        ("Win rate (%)",        "win_rate"),
        ("Trades executed",     "n_trades"),
        ("Trade frequency (%)", "trade_frequency"),
    ]

    agents = [k for k in results if k != "bnh"]
    header = f"  {'Metric':<25}" + "".join(f"{a.upper():>12}" for a in agents) + f"{'BUY&HOLD':>12}"

    print()
    print("=" * (25 + 12 * (len(agents) + 1) + 2))
    print(header)
    print("=" * (25 + 12 * (len(agents) + 1) + 2))

    for label, key in metrics:
        row = f"  {label:<25}"
        for agent in agents:
            val = results[agent].get(key, "N/A")
            row += f"{str(val):>12}"
        bnh_val = results.get("bnh", {}).get(key, "N/A")
        row += f"{str(bnh_val):>12}"
        print(row)

    print("=" * (25 + 12 * (len(agents) + 1) + 2))
    print()


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    TICKER      = "AAPL"
    WINDOW_SIZE = 10
    MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")
    REPORT_DIR  = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print("  Phase 5 — Backtesting on Test Set (2022–2024)")
    print(f"{'='*55}\n")

    # -- Load data -------------------------------------------------------
    test_df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"),
        index_col=0, parse_dates=True,
    )
    raw_all = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "raw", f"{TICKER}.csv"),
        index_col=0, parse_dates=True,
    )
    if isinstance(raw_all.columns, pd.MultiIndex):
        raw_all.columns = raw_all.columns.get_level_values(0)
    test_raw = raw_all.loc[test_df.index]

    print(f"Test period : {test_df.index[0].date()} → {test_df.index[-1].date()}")
    print(f"Test rows   : {len(test_df)}\n")

    results = {}

    # -- DQN backtest ----------------------------------------------------
    dqn_path = os.path.join(MODEL_DIR, f"dqn_{TICKER.lower()}.zip")
    if os.path.exists(dqn_path):
        print("Running DQN on test set...")
        dqn_model = DQN.load(dqn_path)
        dqn_env   = run_backtest(dqn_model, test_df, test_raw, WINDOW_SIZE)
        results["dqn"] = compute_metrics(dqn_env)
        generate_tearsheet(
            dqn_env, test_raw,
            title       = f"DQN Trading Agent — {TICKER} Test Set",
            output_path = os.path.join(REPORT_DIR, "dqn_tearsheet.html"),
        )
    else:
        print(f"[WARN] DQN model not found at {dqn_path}. Train Phase 4 first.")

    # -- PPO backtest ----------------------------------------------------
    ppo_path = os.path.join(MODEL_DIR, f"ppo_{TICKER.lower()}.zip")
    if os.path.exists(ppo_path):
        print("Running PPO on test set...")
        ppo_model = PPO.load(ppo_path)
        ppo_env   = run_backtest(ppo_model, test_df, test_raw, WINDOW_SIZE)
        results["ppo"] = compute_metrics(ppo_env)
        generate_tearsheet(
            ppo_env, test_raw,
            title       = f"PPO Trading Agent — {TICKER} Test Set",
            output_path = os.path.join(REPORT_DIR, "ppo_tearsheet.html"),
        )
    else:
        print(f"[WARN] PPO model not found at {ppo_path}. Train Phase 4 first.")

    # -- Buy & Hold baseline --------------------------------------------
    print("Computing Buy & Hold baseline...")
    results["bnh"] = compute_bnh_metrics(test_raw)

    # -- Print comparison -----------------------------------------------
    if results:
        print_comparison(results)

    print("Phase 5 complete.")
    print(f"Reports saved to: {REPORT_DIR}/")


if __name__ == "__main__":
    main()