"""
agents/ppo_agent.py
-------------------
Trains a PPO agent on the trading environment.

How to run
    python agents/ppo_agent.py

What it produces
    models/ppo_aapl.zip   — saved model weights
    W&B run               — live training curves at wandb.ai
"""

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
from data.pipeline import run_pipeline


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

TICKER         = "AAPL"
TOTAL_STEPS    = 200_000
WINDOW_SIZE    = 10
EVAL_FREQ      = 5_000
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")

PPO_PARAMS = dict(
    learning_rate  = 3e-4,
    n_steps        = 2048,    # steps collected before each update
    batch_size     = 64,
    n_epochs       = 10,      # gradient steps per policy update
    gamma          = 0.99,
    gae_lambda     = 0.95,    # GAE smoothing — reduces variance in advantage estimates
    clip_range     = 0.2,     # PPO clip parameter — core stability mechanism
    ent_coef       = 0.01,    # entropy bonus — encourages exploration
    vf_coef        = 0.5,     # value function loss weight
    max_grad_norm  = 0.5,     # gradient clipping — prevents exploding gradients
    verbose        = 1,
)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def train_ppo(use_wandb: bool = True):

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*50}")
    print("  PPO Agent — Phase 4")
    print(f"{'='*50}\n")

    # -- Load data --------------------------------------------------------
    train_df, val_df, _ = run_pipeline(TICKER)

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
    def make_train_env():
        env = TradingEnv(train_df, train_raw, window_size=WINDOW_SIZE)
        return Monitor(env)

    def make_val_env():
        env = TradingEnv(val_df, val_raw, window_size=WINDOW_SIZE)
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
                name    = f"ppo_{TICKER}",
                config  = PPO_PARAMS,
                sync_tensorboard = True,
            )
            callbacks.append(
                WandbCallback(
                    gradient_save_freq = 1_000,
                    model_save_path    = os.path.join(MODEL_SAVE_DIR, "ppo_checkpoints"),
                    verbose            = 1,
                )
            )
            print("W&B tracking enabled.")
        except ImportError:
            print("[WARN] wandb not installed — training without W&B tracking.")

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals = 10,
        min_evals                = 20,
        verbose                  = 1,
    )
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path = os.path.join(MODEL_SAVE_DIR, "ppo_best"),
        log_path             = os.path.join(MODEL_SAVE_DIR, "ppo_logs"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 3,
        deterministic        = True,
        callback_after_eval  = stop_callback,
        verbose              = 1,
    )
    callbacks.append(eval_callback)

    # -- Train ------------------------------------------------------------
    model = PPO(
        policy = "MlpPolicy",
        env    = train_env,
        tensorboard_log = os.path.join(MODEL_SAVE_DIR, "ppo_tensorboard"),
        **PPO_PARAMS,
    )

    print(f"\nTraining PPO for {TOTAL_STEPS:,} steps...\n")
    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = callbacks,
        progress_bar    = True,
    )

    # -- Save -------------------------------------------------------------
    save_path = os.path.join(MODEL_SAVE_DIR, f"ppo_{TICKER.lower()}")
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
    train_ppo(use_wandb=True)