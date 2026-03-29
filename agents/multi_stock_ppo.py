"""
agents/multi_stock_ppo.py
--------------------------
Trains a generalised PPO agent across 8 diverse S&P 500 stocks simultaneously.

Strategy
--------
- Build one TradingEnv per stock → wrap all in DummyVecEnv (8 parallel envs)
- Warm-start from the existing ppo_aapl.zip checkpoint (avoids training from scratch)
- PPO collects n_steps from every env simultaneously → 8x more diverse experience
  per gradient update compared to single-stock training
- Early stopping on AAPL validation set (stable, comparable to Phase 4 baseline)
- Output: models/ppo_multi.zip

How to run
    # Fast sanity check (~15 min on CPU)
    python agents/multi_stock_ppo.py --steps 100000

    # Full training run (~30-60 min on CPU)
    python agents/multi_stock_ppo.py --steps 300000

    # Skip warm-start, train from scratch
    python agents/multi_stock_ppo.py --no-warmstart

    # Disable W&B logging
    python agents/multi_stock_ppo.py --no-wandb
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from env.trading_env import TradingEnv
from data.pipeline import fetch_and_process


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

# 8 tickers chosen for sector diversity and long price history (all pre-2013)
TICKERS = [
    "AAPL",   # Tech / Consumer Electronics  — existing training baseline
    "MSFT",   # Tech / Enterprise            — similar sector, different dynamics
    "TSLA",   # Growth / High-Volatility     — extreme swings, stress test
    "JPM",    # Financial                    — rate-sensitive
    "JNJ",    # Defensive Healthcare         — low volatility, slow moving
    "XOM",    # Energy                       — commodity-driven, AAPL anti-correlated
    "AMZN",   # Consumer / Cloud             — long bull run + COVID crash
    "PG",     # Consumer Staples             — very defensive, smoothest equity curve
]

WINDOW_SIZE    = 10
EVAL_FREQ      = 5_000       # evaluate on val env every N steps
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
WARMSTART_PATH = os.path.join(MODEL_SAVE_DIR, "ppo_aapl.zip")
OUTPUT_NAME    = "ppo_multi"

# PPO hyperparams — keep most the same as single-stock run.
# Lower LR slightly for fine-tuning stability when warm-starting.
PPO_PARAMS = dict(
    learning_rate  = 1e-4,    # slightly lower than 3e-4 for warm-start fine-tuning
    n_steps        = 2048,    # steps per env before each update
    batch_size     = 128,     # larger batch to benefit from 8× env diversity
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    verbose        = 1,
)


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load_all_stocks(tickers: list[str]) -> dict:
    """
    Download + feature-engineer + scale all tickers.
    Returns dict: ticker -> (train_df, val_df, test_df, raw_df)
    Uses on-disk cache in data/raw/ so subsequent runs are instant.
    """
    data = {}
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] Loading {ticker}...", end=" ", flush=True)
        try:
            train_df, val_df, test_df, raw_df = fetch_and_process(ticker)
            data[ticker] = (train_df, val_df, test_df, raw_df)
            print(f"OK  ({len(train_df)} train rows, {len(val_df)} val rows)")
        except Exception as e:
            print(f"FAILED — {e}")
            print(f"         Skipping {ticker}. Training will continue with remaining stocks.")
    return data


# ------------------------------------------------------------------ #
# Environment factories
# ------------------------------------------------------------------ #

def make_train_env(ticker: str, train_df: pd.DataFrame, raw_df: pd.DataFrame):
    """Return a callable that creates a monitored TradingEnv for one training stock."""
    def _make():
        env = TradingEnv(train_df, raw_df.loc[train_df.index], window_size=WINDOW_SIZE,
                         reward_mode="sharpe")
        return Monitor(env)
    return _make


def make_val_env(val_df: pd.DataFrame, raw_df: pd.DataFrame):
    """Return a callable that creates the AAPL validation environment."""
    def _make():
        env = TradingEnv(val_df, raw_df.loc[val_df.index], window_size=WINDOW_SIZE,
                         reward_mode="sharpe")
        return Monitor(env)
    return _make


# ------------------------------------------------------------------ #
# Main training function
# ------------------------------------------------------------------ #

def train_multi_stock_ppo(total_steps: int = 300_000,
                          warm_start: bool = True,
                          use_wandb: bool = True) -> PPO:

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print("  Multi-Stock PPO — Generalised Trading Agent")
    print(f"{'='*55}\n")
    print(f"  Tickers     : {', '.join(TICKERS)}")
    print(f"  Steps       : {total_steps:,}")
    print(f"  Warm-start  : {'YES — loading ' + WARMSTART_PATH if warm_start else 'NO — training from scratch'}")
    print()

    # -- Load data for all tickers ----------------------------------------
    print("Loading and processing stock data...")
    stock_data = load_all_stocks(TICKERS)

    if len(stock_data) == 0:
        raise RuntimeError("No stocks loaded successfully. Check your internet connection.")

    loaded_tickers = list(stock_data.keys())
    print(f"\n  Successfully loaded {len(loaded_tickers)}/{len(TICKERS)} stocks: {', '.join(loaded_tickers)}\n")

    # -- Build vectorised training environment (one env per stock) --------
    train_env_fns = [
        make_train_env(t, stock_data[t][0], stock_data[t][3])
        for t in loaded_tickers
    ]
    train_env = DummyVecEnv(train_env_fns)

    n_envs = len(train_env_fns)
    print(f"  Training with {n_envs} parallel environments")
    print(f"  Steps per gradient update: {PPO_PARAMS['n_steps'] * n_envs:,}")
    print(f"  Effective training rows seen: ~{sum(len(stock_data[t][0]) for t in loaded_tickers):,}\n")

    # -- Build AAPL validation environment --------------------------------
    # Use AAPL val either from fetched data or fall back to existing processed splits
    if "AAPL" in stock_data:
        aapl_val_df  = stock_data["AAPL"][1]
        aapl_raw_df  = stock_data["AAPL"][3]
    else:
        # Fallback: load the original processed val split
        print("  [WARN] AAPL not in stock_data — loading val split from data/processed/")
        aapl_val_df = pd.read_csv(
            os.path.join(PROJECT_ROOT, "data", "processed", "val.csv"),
            index_col=0, parse_dates=True,
        )
        aapl_raw_df = pd.read_csv(
            os.path.join(PROJECT_ROOT, "data", "raw", "AAPL.csv"),
            index_col=0, parse_dates=True,
        )

    val_env = DummyVecEnv([make_val_env(aapl_val_df, aapl_raw_df)])

    # -- W&B setup --------------------------------------------------------
    callbacks = []

    if use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            run = wandb.init(
                project = "rl-trading-agent",
                name    = f"ppo_multi_{n_envs}stocks_{total_steps//1000}k",
                config  = {**PPO_PARAMS, "tickers": loaded_tickers, "total_steps": total_steps},
                sync_tensorboard = True,
            )
            callbacks.append(
                WandbCallback(
                    gradient_save_freq = 2_000,
                    model_save_path    = os.path.join(MODEL_SAVE_DIR, "ppo_multi_checkpoints"),
                    verbose            = 1,
                )
            )
            print("  W&B tracking enabled.\n")
        except ImportError:
            print("  [WARN] wandb not installed — training without W&B tracking.\n")

    # Early stopping: stop if AAPL val reward hasn't improved for 10 evals
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals = 10,
        min_evals                = 15,
        verbose                  = 1,
    )
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path = os.path.join(MODEL_SAVE_DIR, "ppo_multi_best"),
        log_path             = os.path.join(MODEL_SAVE_DIR, "ppo_multi_logs"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 3,
        deterministic        = True,
        callback_after_eval  = stop_callback,
        verbose              = 1,
    )
    callbacks.append(eval_callback)

    # -- Build or load model ----------------------------------------------
    if warm_start and os.path.exists(WARMSTART_PATH):
        print(f"  Warm-starting from {WARMSTART_PATH}")
        print(f"  (Loading existing AAPL policy weights — will fine-tune on all stocks)\n")
        model = PPO.load(
            WARMSTART_PATH,
            env    = train_env,
            # Override LR and clip for fine-tuning
            custom_objects = {
                "learning_rate": PPO_PARAMS["learning_rate"],
                "clip_range":    PPO_PARAMS["clip_range"],
            },
            verbose = 1,
        )
        # Update batch size and n_epochs via set_parameters isn't needed —
        # these are used by .learn() automatically from the loaded model's policy
    else:
        if warm_start and not os.path.exists(WARMSTART_PATH):
            print(f"  [WARN] Warm-start requested but {WARMSTART_PATH} not found.")
            print(f"         Training from scratch instead.\n")
        else:
            print("  Training from scratch (no warm-start).\n")

        model = PPO(
            policy         = "MlpPolicy",
            env            = train_env,
            tensorboard_log = os.path.join(MODEL_SAVE_DIR, "ppo_multi_tensorboard"),
            **PPO_PARAMS,
        )

    # -- Train ------------------------------------------------------------
    print(f"  Starting training for {total_steps:,} steps...\n")
    model.learn(
        total_timesteps = total_steps,
        callback        = callbacks,
        progress_bar    = True,
        reset_num_timesteps = not warm_start,  # continue step counter if warm-starting
    )

    # -- Save -------------------------------------------------------------
    save_path = os.path.join(MODEL_SAVE_DIR, OUTPUT_NAME)
    model.save(save_path)
    print(f"\n  Model saved → {save_path}.zip")

    # -- Quick validation -------------------------------------------------
    print("\n  Running final evaluation on AAPL validation set (5 episodes)...")
    mean_reward = _evaluate(model, val_env, n_episodes=5)
    print(f"  Val mean reward: {mean_reward:.4f}")

    # Also evaluate on each training stock's val split for cross-stock check
    print("\n  Cross-stock validation (1 episode each):")
    print(f"  {'Ticker':<8} {'Val Return':>12} {'Trades':>8}")
    print(f"  {'-'*30}")
    for ticker in loaded_tickers:
        try:
            v_df  = stock_data[ticker][1]
            r_df  = stock_data[ticker][3]
            if len(v_df) == 0:
                continue
            # Wrap in DummyVecEnv so model.predict() gets the right batch shape
            eval_env_single = DummyVecEnv([make_val_env(v_df, r_df)])
            obs_v  = eval_env_single.reset()
            done_v = False
            while not done_v:
                action_v, _ = model.predict(obs_v, deterministic=True)
                obs_v, _, done_v, _ = eval_env_single.step(action_v)
            # Extract portfolio history from the underlying env
            inner_env = eval_env_single.envs[0].env
            port = pd.Series(inner_env.portfolio_history)
            ret  = (port.iloc[-1] / 10_000 - 1) * 100
            n_tr = len(inner_env.get_trade_log())
            print(f"  {ticker:<8} {ret:>+11.2f}%  {n_tr:>7d}")
        except Exception as e:
            print(f"  {ticker:<8} ERROR: {e}")

    if use_wandb:
        try:
            wandb.log({"final_val_mean_reward": mean_reward})
            wandb.finish()
        except Exception:
            pass

    print(f"\n{'='*55}")
    print(f"  Training complete. Model saved to models/{OUTPUT_NAME}.zip")
    print(f"  Open the dashboard and select 'Multi-stock' model to use it.")
    print(f"{'='*55}\n")

    return model


# ------------------------------------------------------------------ #
# Evaluation helper
# ------------------------------------------------------------------ #

def _evaluate(model: PPO, vec_env: DummyVecEnv, n_episodes: int = 5) -> float:
    rewards = []
    for _ in range(n_episodes):
        obs    = vec_env.reset()
        ep_rew = 0.0
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            ep_rew += float(reward[0])
        rewards.append(ep_rew)
    return float(np.mean(rewards))


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a multi-stock generalised PPO trading agent."
    )
    parser.add_argument(
        "--steps", type=int, default=300_000,
        help="Total training timesteps. Use 100000 for a quick sanity check (default: 300000)."
    )
    parser.add_argument(
        "--no-warmstart", action="store_true",
        help="Train from scratch instead of loading ppo_aapl.zip as starting point."
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging."
    )
    args = parser.parse_args()

    train_multi_stock_ppo(
        total_steps = args.steps,
        warm_start  = not args.no_warmstart,
        use_wandb   = not args.no_wandb,
    )
