"""
env/trading_env.py
------------------
Custom Gymnasium environment for the RL trading agent.

MDP formulation
    State  : sliding window of (window_size, n_features) normalised features
    Action : Discrete(3)  —  0=Hold  1=Buy (10% of cash)  2=Sell (all shares)
    Reward : risk-adjusted (default) or raw percentage return

Reward modes
    "sharpe"  (default)
        Combines a rolling Sharpe ratio component with a drawdown penalty.
        Encourages the agent to seek consistent, low-volatility returns and
        to exit positions quickly when the portfolio starts losing ground.

        reward = Sharpe_component + λ_dd × drawdown_penalty

        Sharpe_component = mean(R_window) / (std(R_window) + ε)
            where R_window is a rolling deque of the last `sharpe_window`
            per-step percentage returns.

        drawdown_penalty = min(0, (value − peak) / peak)
            Always ≤ 0; zero while the portfolio is at or above its peak.

    "raw"
        Original v1 reward: simple per-step percentage return.
        r_t = (V_t − V_{t-1}) / V_{t-1}
        Use this to reproduce earlier training runs or during pure inference.

FIX (v2)
    The environment now accepts a separate `raw_df` for real dollar prices.
    `df`     → normalised features fed to the agent as observations
    `raw_df` → raw (un-scaled) OHLCV used only for trade execution & reward

    This prevents the RobustScaler from producing negative Close values
    that break share counts and portfolio calculations.

Usage
    from env.trading_env import TradingEnv

    # New default — risk-adjusted reward (use when retraining)
    env = TradingEnv(df=train_scaled, raw_df=train_raw)

    # Original raw reward (backward compatible, use for inference only)
    env = TradingEnv(df=train_scaled, raw_df=train_raw, reward_mode="raw")
"""

from collections import deque

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
    reward_mode : str
        "sharpe" (default) — rolling Sharpe + drawdown penalty.
        "raw"              — simple per-step percentage return (v1 behaviour).
    sharpe_window : int
        Number of most-recent step-returns used to compute the rolling Sharpe.
        Default 20 (~one trading month).
    dd_lambda : float
        Weight of the drawdown penalty term in the Sharpe reward.
        Default 0.5.  Set to 0.0 to disable the penalty entirely.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        raw_df: pd.DataFrame,
        initial_balance: float  = 10_000.0,
        transaction_cost: float = 0.001,
        window_size: int        = 10,
        reward_mode: str        = "sharpe",
        sharpe_window: int      = 20,
        dd_lambda: float        = 0.5,
    ):
        super().__init__()

        assert len(df) == len(raw_df), (
            f"df and raw_df must have the same number of rows. "
            f"Got df={len(df)}, raw_df={len(raw_df)}"
        )
        assert "Close" in raw_df.columns, (
            "raw_df must contain a 'Close' column with real dollar prices."
        )
        assert reward_mode in ("sharpe", "raw"), (
            f"reward_mode must be 'sharpe' or 'raw', got '{reward_mode}'."
        )

        self.df               = df.reset_index(drop=True)       # normalised → observations
        self.raw_df           = raw_df.reset_index(drop=True)   # raw prices → trading logic
        self.initial_balance  = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size      = window_size
        self.n_features       = df.shape[1]
        self.reward_mode      = reward_mode
        self.sharpe_window    = sharpe_window
        self.dd_lambda        = dd_lambda

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

        # Risk-adjusted reward state (only used when reward_mode="sharpe")
        self._return_window = deque(maxlen=self.sharpe_window)
        self._peak_value    = initial_balance

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

        # Reset risk-tracking state
        self._return_window = deque(maxlen=self.sharpe_window)
        self._peak_value    = self.initial_balance

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

        reward = self._compute_reward(prev_value, current_value)

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
    # Reward computation
    # ------------------------------------------------------------------ #

    def _compute_reward(self, prev_value: float, current_value: float) -> float:
        """
        Compute the per-step reward based on `self.reward_mode`.

        Sharpe mode
        -----------
        step_return  r_t = (V_t - V_{t-1}) / V_{t-1}

        Rolling Sharpe component
            Appended to a fixed-length deque of recent returns.
            Once the window has ≥ 5 observations:
                sharpe_r = mean(window) / (std(window) + ε)
            This rewards *consistent* positive returns and penalises noisy,
            high-variance trading (e.g. buying and selling every day with
            small gains that don't offset transaction costs).

        Drawdown penalty
            peak  ← running maximum portfolio value seen so far
            dd    = (V_t - peak) / peak      (≤ 0 when below peak, 0 otherwise)
            penalty = λ_dd × min(0, dd)

            Effect: the agent is nudged to exit positions when the portfolio
            starts sliding below its high-water mark, rather than holding
            through large drawdowns hoping for a recovery.

        Combined
            reward = sharpe_r + λ_dd × min(0, drawdown)

        Raw mode
        --------
        reward = r_t  (original v1 behaviour — simple percentage return)
        """
        # Per-step percentage return (used in both modes for the Sharpe window)
        step_return = (current_value - prev_value) / (prev_value + 1e-8)

        if self.reward_mode == "raw":
            return float(step_return)

        # ---- Sharpe mode ------------------------------------------------

        # 1. Update rolling return window
        self._return_window.append(step_return)

        # 2. Update high-water mark (peak portfolio value)
        self._peak_value = max(self._peak_value, current_value)

        # 3. Rolling Sharpe component
        #    Use raw step_return until the window has enough observations.
        if len(self._return_window) >= 5:
            arr      = np.array(self._return_window, dtype=np.float64)
            mu       = arr.mean()
            sigma    = arr.std() + 1e-8
            sharpe_r = float(mu / sigma)
        else:
            sharpe_r = float(step_return)

        # 4. Drawdown penalty (always ≤ 0)
        drawdown   = (current_value - self._peak_value) / (self._peak_value + 1e-8)
        dd_penalty = min(0.0, float(drawdown))

        return sharpe_r + self.dd_lambda * dd_penalty

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