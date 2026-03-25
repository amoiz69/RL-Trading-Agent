"""
env/trading_env.py
------------------
Custom Gymnasium environment for the RL trading agent.

MDP formulation
    State  : sliding window of (window_size, n_features) normalised features
    Action : Discrete(3)  —  0=Hold  1=Buy (10% of cash)  2=Sell (all shares)
    Reward : percentage change in total portfolio value per step

FIX (v2)
    The environment now accepts a separate `raw_df` for real dollar prices.
    `df`     → normalised features fed to the agent as observations
    `raw_df` → raw (un-scaled) OHLCV used only for trade execution & reward

    This prevents the RobustScaler from producing negative Close values
    that break share counts and portfolio calculations.

Usage
    from env.trading_env import TradingEnv

    env = TradingEnv(df=train_scaled, raw_df=train_raw)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Single-stock trading environment compatible with Stable-Baselines3.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised feature DataFrame (output of RobustScaler in pipeline.py).
        Shape: (n_timesteps, n_features). Used only for observations.
    raw_df : pd.DataFrame
        Raw, un-scaled OHLCV DataFrame. Must contain a 'Close' column with
        real dollar prices. Used only for trade execution and reward.
        Must have the same number of rows and index as `df`.
    initial_balance : float
        Starting cash in USD. Default $10,000.
    transaction_cost : float
        Fraction of trade value charged as commission. Default 0.1% (0.001).
    window_size : int
        Number of past timesteps in each observation. Default 10.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        raw_df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.001,
        window_size: int = 10,
    ):
        super().__init__()

        assert len(df) == len(raw_df), (
            f"df and raw_df must have the same number of rows. "
            f"Got df={len(df)}, raw_df={len(raw_df)}"
        )
        assert "Close" in raw_df.columns, (
            "raw_df must contain a 'Close' column with real dollar prices."
        )

        self.df               = df.reset_index(drop=True)       # normalised → observations
        self.raw_df           = raw_df.reset_index(drop=True)   # raw prices → trading logic
        self.initial_balance  = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size      = window_size
        self.n_features       = df.shape[1]

        # -- Spaces -------------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)   # 0=Hold  1=Buy  2=Sell

        # -- Internal state (reset in reset()) ----------------------------
        self.balance           = initial_balance
        self.shares            = 0.0
        self.current_step      = window_size
        self.portfolio_history = []
        self.trade_log         = []

    # ------------------------------------------------------------------ #
    # Core Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance           = self.initial_balance
        self.shares            = 0.0
        self.current_step      = self.window_size
        self.portfolio_history = [self.initial_balance]
        self.trade_log         = []

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Execute one timestep.
        Returns: obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_price = self._current_price()           # real dollar price
        prev_value    = self._portfolio_value(current_price)

        # -- Execute trade ------------------------------------------------
        if action == 1:    # Buy — spend 10% of available cash
            cash_to_spend = self.balance * 0.10
            if cash_to_spend > 0:
                shares_bought = cash_to_spend * (1 - self.transaction_cost) / current_price
                self.shares  += shares_bought
                self.balance -= cash_to_spend
                self.trade_log.append({
                    "step":   self.current_step,
                    "action": "BUY",
                    "price":  round(current_price, 4),
                    "shares": round(shares_bought, 6),
                    "cost":   round(cash_to_spend * self.transaction_cost, 4),
                })

        elif action == 2:  # Sell — liquidate all shares
            if self.shares > 0:
                proceeds = self.shares * current_price * (1 - self.transaction_cost)
                self.trade_log.append({
                    "step":   self.current_step,
                    "action": "SELL",
                    "price":  round(current_price, 4),
                    "shares": round(self.shares, 6),
                    "cost":   round(self.shares * current_price * self.transaction_cost, 4),
                })
                self.balance += proceeds
                self.shares   = 0.0

        # action == 0 → Hold, no state change

        # -- Advance time -------------------------------------------------
        self.current_step += 1

        # -- Compute reward -----------------------------------------------
        current_price = self._current_price()           # price at new step
        current_value = self._portfolio_value(current_price)
        reward        = (current_value - prev_value) / (prev_value + 1e-8)

        self.portfolio_history.append(current_value)

        # -- Termination --------------------------------------------------
        terminated = self.current_step >= len(self.df) - 1
        truncated  = False

        obs  = self._get_obs()
        info = {
            "portfolio_value": round(current_value, 2),
            "balance":         round(self.balance,  2),
            "shares":          round(self.shares,   6),
            "price":           round(current_price, 4),
            "step":            self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        price = self._current_price()
        value = self._portfolio_value(price)
        print(
            f"Step {self.current_step:>5} | "
            f"Price ${price:>8.2f} | "
            f"Balance ${self.balance:>10.2f} | "
            f"Shares {self.shares:>8.4f} | "
            f"Portfolio ${value:>10.2f}"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        """Slice the normalised feature window. Shape: (window_size, n_features)."""
        return self.df.iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)

    def _current_price(self) -> float:
        """Real dollar Close price from raw_df at current_step."""
        idx   = min(self.current_step, len(self.raw_df) - 1)
        price = float(self.raw_df.iloc[idx]["Close"])
        assert price > 0, (
            f"Raw price at step {idx} is {price:.4f}. "
            "Ensure raw_df contains un-scaled dollar prices, not normalised values."
        )
        return price

    def _portfolio_value(self, price: float) -> float:
        """Total value = cash balance + market value of shares held."""
        return self.balance + self.shares * price

    # ------------------------------------------------------------------ #
    # Convenience methods (used by backtester & dashboard)
    # ------------------------------------------------------------------ #

    def get_trade_log(self) -> pd.DataFrame:
        """All executed trades as a DataFrame."""
        return pd.DataFrame(self.trade_log)

    def get_portfolio_series(self) -> pd.Series:
        """Portfolio value history aligned to the DataFrame's date index."""
        idx = self.df.index[
            self.window_size : self.window_size + len(self.portfolio_history)
        ]
        return pd.Series(self.portfolio_history, index=idx, name="portfolio_value")