"""
agents/dqn_agent.py
-------------------
Trains a DQN agent on the trading environment.

Uses risk-adjusted reward (rolling Sharpe + drawdown penalty) by default.
Pass reward_mode='raw' to TradingEnv to reproduce original v1 behaviour.

How to run
    python agents/dqn_agent.py

What it produces
    models/dqn_aapl.zip   — saved model weights
    W&B run               — live training curves at wandb.ai
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from env.trading_env import TradingEnv
from data.pipeline import run_pipeline


# ------------------------------------------------------------------ #
# Config — change these to experiment
# ------------------------------------------------------------------ #

TICKER         = "AAPL"
TOTAL_STEPS    = 200_000
WINDOW_SIZE    = 10
EVAL_FREQ      = 5_000       # evaluate on val env every N steps
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")

DQN_PARAMS = dict(
    learning_rate           = 1e-4,
    buffer_size             = 50_000,
    learning_starts         = 1_000,   # fill buffer before training starts
    batch_size              = 64,
    gamma                   = 0.99,    # discount factor
    train_freq              = 4,       # update every 4 steps
    target_update_interval  = 500,     # copy to target network every 500 steps
    exploration_fraction    = 0.15,    # first 15% of steps = random exploration
    exploration_final_eps   = 0.05,    # minimum exploration rate
    verbose                 = 1,
)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def train_dqn(use_wandb: bool = True):

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # -- Load data --------------------------------------------------------
    print(f"\n{'='*50}")
    print("  DQN Agent — Phase 4")
    print(f"{'='*50}\n")

    train_df, val_df, _ = run_pipeline(TICKER)

    # Raw un-scaled prices for trade execution
    raw_all = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "raw", f"{TICKER}.csv"),
        index_col=0, parse_dates=True,
    )
    if isinstance(raw_all.columns, pd.MultiIndex):
        raw_all.columns = raw_all.columns.get_level_values(0)

    train_raw = raw_all.loc[train_df.index]
    val_raw   = raw_all.loc[val_df.index]

    print(f"Train : {len(train_df)} rows  |  Val : {len(val_df)} rows")

    # -- Build environments -----------------------------------------------
    # Training env — wrapped in Monitor for SB3 episode logging
    def make_train_env():
        env = TradingEnv(train_df, train_raw, window_size=WINDOW_SIZE,
                         reward_mode="sharpe")
        return Monitor(env)

    def make_val_env():
        env = TradingEnv(val_df, val_raw, window_size=WINDOW_SIZE,
                         reward_mode="sharpe")
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])
    val_env   = DummyVecEnv([make_val_env])

    # -- W&B setup --------------------------------------------------------
    callbacks = []

    if use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback

            run = wandb.init(
                project = "rl-trading-agent",
                name    = f"dqn_{TICKER}",
                config  = DQN_PARAMS,
                sync_tensorboard = True,
            )
            callbacks.append(
                WandbCallback(
                    gradient_save_freq = 1_000,
                    model_save_path    = os.path.join(MODEL_SAVE_DIR, "dqn_checkpoints"),
                    verbose            = 1,
                )
            )
            print("W&B tracking enabled.")
        except ImportError:
            print("[WARN] wandb not installed — training without W&B tracking.")
            print("       Install with: pip install wandb")

    # EvalCallback — saves the best model checkpoint automatically
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals = 10,
        min_evals                = 20,
        verbose                  = 1,
    )
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path = os.path.join(MODEL_SAVE_DIR, "dqn_best"),
        log_path             = os.path.join(MODEL_SAVE_DIR, "dqn_logs"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 3,
        deterministic        = True,
        callback_after_eval  = stop_callback,
        verbose              = 1,
    )
    callbacks.append(eval_callback)

    # -- Train ------------------------------------------------------------
    model = DQN(
        policy = "MlpPolicy",
        env    = train_env,
        tensorboard_log = os.path.join(MODEL_SAVE_DIR, "dqn_tensorboard"),
        **DQN_PARAMS,
    )

    print(f"\nTraining DQN for {TOTAL_STEPS:,} steps...\n")
    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = callbacks,
        progress_bar    = True,
    )

    # -- Save -------------------------------------------------------------
    save_path = os.path.join(MODEL_SAVE_DIR, f"dqn_{TICKER.lower()}")
    model.save(save_path)
    print(f"\nModel saved → {save_path}.zip")

    # -- Quick val evaluation ---------------------------------------------
    mean_reward = _evaluate(model, val_env, n_episodes=5)
    print(f"Val mean reward (5 episodes): {mean_reward:.4f}")

    if use_wandb:
        try:
            wandb.log({"val/mean_reward": mean_reward})
            wandb.finish()
        except Exception:
            pass

    return model


def _evaluate(model, vec_env, n_episodes: int = 5) -> float:
    """Run n_episodes with the trained model and return mean episode reward."""
    rewards = []
    for _ in range(n_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            episode_reward += reward[0]
        rewards.append(episode_reward)
    return float(np.mean(rewards))


if __name__ == "__main__":
    train_dqn(use_wandb=True)