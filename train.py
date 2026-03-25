"""
train.py
--------
Top-level training script. Trains both DQN and PPO sequentially
and prints a comparison table at the end.

How to run
    python train.py                    # train both agents with W&B
    python train.py --no-wandb         # train without W&B logging
    python train.py --agent dqn        # train only DQN
    python train.py --agent ppo        # train only PPO
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env.trading_env import TradingEnv
from data.pipeline import run_pipeline
from agents.dqn_agent import train_dqn
from agents.ppo_agent import train_ppo


def evaluate_agent(model, df, raw_df, window_size=10, initial_balance=10_000):
    """
    Run one full deterministic episode and return a metrics dict.
    Called after training to compare DQN vs PPO on the val set.
    """
    env = TradingEnv(df, raw_df, initial_balance=initial_balance,
                     window_size=window_size)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    portfolio   = pd.Series(env.portfolio_history)
    returns     = portfolio.pct_change().dropna()
    trade_log   = env.get_trade_log()

    final_value    = portfolio.iloc[-1]
    total_return   = (final_value - initial_balance) / initial_balance * 100
    sharpe         = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
    rolling_max    = portfolio.cummax()
    drawdown       = ((portfolio - rolling_max) / rolling_max)
    max_drawdown   = drawdown.min() * 100

    n_trades   = len(trade_log)
    if n_trades > 0 and 'action' in trade_log.columns:
        sells      = trade_log[trade_log['action'] == 'SELL']
        buys       = trade_log[trade_log['action'] == 'BUY']
        win_rate   = _calc_win_rate(buys, sells, df)
    else:
        win_rate   = 0.0

    return {
        "final_value"  : round(final_value,  2),
        "total_return" : round(total_return, 2),
        "sharpe"       : round(float(sharpe), 3),
        "max_drawdown" : round(float(max_drawdown), 2),
        "n_trades"     : n_trades,
        "win_rate"     : round(win_rate, 1),
    }


def _calc_win_rate(buys, sells, df):
    """Rough win rate: fraction of sell prices higher than average buy price."""
    if len(buys) == 0 or len(sells) == 0:
        return 0.0
    avg_buy  = buys['price'].mean()
    wins     = (sells['price'] > avg_buy).sum()
    return wins / len(sells) * 100


def print_comparison(results: dict):
    """Print a formatted comparison table."""
    metrics = ["final_value", "total_return", "sharpe", "max_drawdown",
               "n_trades",    "win_rate"]
    labels  = ["Final portfolio ($)", "Total return (%)", "Sharpe ratio",
               "Max drawdown (%)",   "Trades executed",  "Win rate (%)"]

    print("\n" + "=" * 60)
    print(f"  {'Metric':<25} {'DQN':>10} {'PPO':>10}")
    print("=" * 60)
    for metric, label in zip(metrics, labels):
        dqn_val = results.get("dqn", {}).get(metric, "N/A")
        ppo_val = results.get("ppo", {}).get(metric, "N/A")
        print(f"  {label:<25} {str(dqn_val):>10} {str(ppo_val):>10}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",    choices=["dqn", "ppo", "both"],
                        default="both")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    # Load data once — reused for evaluation
    print("\nLoading data...")
    train_df, val_df, _ = run_pipeline("AAPL")

    raw_all = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "raw", "AAPL.csv"),
        index_col=0, parse_dates=True,
    )
    if isinstance(raw_all.columns, pd.MultiIndex):
        raw_all.columns = raw_all.columns.get_level_values(0)
    val_raw = raw_all.loc[val_df.index]

    results = {}

    # Train DQN
    if args.agent in ("dqn", "both"):
        dqn_model = train_dqn(use_wandb=use_wandb)
        print("\nEvaluating DQN on validation set...")
        results["dqn"] = evaluate_agent(dqn_model, val_df, val_raw)

    # Train PPO
    if args.agent in ("ppo", "both"):
        ppo_model = train_ppo(use_wandb=use_wandb)
        print("\nEvaluating PPO on validation set...")
        results["ppo"] = evaluate_agent(ppo_model, val_df, val_raw)

    # Print comparison
    if len(results) == 2:
        print_comparison(results)

    print("Phase 4 complete. Models saved to models/")
    print("Next step: Phase 5 — backtesting on the test set.")


if __name__ == "__main__":
    main()