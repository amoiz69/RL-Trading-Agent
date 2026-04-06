"""
data/pipeline.py
----------------
Data download, feature engineering, and preprocessing pipeline.

Functions
---------
download(ticker, start, end)
    Download raw OHLCV from yfinance (cached to data/raw/).

run_pipeline(ticker)
    Load the pre-built train/val/test splits from data/processed/.
    Used by train.py and the original notebooks (AAPL fixed splits).

fetch_and_process(ticker, start, end)
    Full dynamic pipeline for any ticker:
        1. Download raw OHLCV (cached)
        2. Engineer the same 8 features used during AAPL training
           (Close, rsi, macd, ema_20, ema_50, bb_width, obv, atr)
        3. Fit a fresh RobustScaler on the train portion
        4. Return (train_df, val_df, test_df, raw_df)
    Used by the multi-stock Streamlit dashboard.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Date splits — must match AAPL training splits so observations are compatible
TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END   = "2021-12-31"
TEST_START = "2022-01-01"


# ------------------------------------------------------------------ #
# Raw data download
# ------------------------------------------------------------------ #

def download(ticker: str = "AAPL",
             start: str = "2013-01-01",
             end: str   = "2024-12-31") -> pd.DataFrame:
    """
    Download raw OHLCV (or load from disk if already present).

    IMPORTANT: Paths are resolved relative to the project root, not the
    current working directory (so notebooks work even when `cwd=notebooks/`).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.csv"

    # Validate any existing cache — re-download if file is too small (likely corrupt)
    if path.exists() and path.stat().st_size < 1024:
        print(f"    [WARN] {path.name} looks corrupt ({path.stat().st_size} bytes). Re-downloading...")
        path.unlink()

    if not path.exists():
        df = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=False
        )
        if df is None or len(df) == 0:
            raise ValueError(
                f"yfinance returned no data for '{ticker}'. "
                "Check the ticker symbol or your internet connection."
            )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(path)

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ------------------------------------------------------------------ #
# Feature engineering — identical to AAPL training feature set
# ------------------------------------------------------------------ #

def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _compute_macd(prices: pd.Series,
                  fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast   = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow   = prices.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line          # MACD histogram


def _compute_bb_width(prices: pd.Series, period: int = 20) -> pd.Series:
    sma  = prices.rolling(period).mean()
    std  = prices.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / (sma + 1e-8)


def _compute_atr(high: pd.Series, low: pd.Series,
                 close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 8 features used during AAPL training:
        Close, rsi, macd, ema_20, ema_50, bb_width, obv, atr
    Returns a DataFrame with NaN rows dropped.
    """
    df = pd.DataFrame(index=raw.index)
    close  = raw["Close"].astype(float)
    high   = raw["High"].astype(float)
    low    = raw["Low"].astype(float)
    volume = raw["Volume"].astype(float)

    df["Close"]    = close
    df["rsi"]      = _compute_rsi(close)
    df["macd"]     = _compute_macd(close)
    df["ema_20"]   = close.ewm(span=20, adjust=False).mean()
    df["ema_50"]   = close.ewm(span=50, adjust=False).mean()
    df["bb_width"] = _compute_bb_width(close)
    df["obv"]      = (np.sign(close.diff()) * volume).cumsum()
    df["atr"]      = _compute_atr(high, low, close)

    return df.dropna()


# ------------------------------------------------------------------ #
# Original fixed-split loader (used by train.py and notebooks)
# ------------------------------------------------------------------ #

def run_pipeline(ticker: str = "AAPL"):
    """
    Load preprocessed train/val/test splits and ensure raw data exists.

    The notebooks in this repo expect:
      - `train_df`, `val_df`, `test_df` from `data/processed/*.csv`
      - raw OHLCV from `data/raw/{ticker}.csv` (loaded separately in notebooks)

    Notes
    -----
    Feature engineering is assumed to have been done already and written to:
      - `data/processed/train.csv`
      - `data/processed/val.csv`
      - `data/processed/test.csv`
    """
    # Ensure raw file exists (so notebooks can load it in addition to scaled features).
    download(ticker)

    train_path = PROCESSED_DIR / "train.csv"
    val_path   = PROCESSED_DIR / "val.csv"
    test_path  = PROCESSED_DIR / "test.csv"

    missing = [str(p) for p in (train_path, val_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed splits are missing. Expected these files:\n"
            f"  - data/processed/train.csv\n"
            f"  - data/processed/val.csv\n"
            f"  - data/processed/test.csv\n"
            f"\nMissing on disk: {missing}"
        )

    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    val_df   = pd.read_csv(val_path,   index_col=0, parse_dates=True)
    test_df  = pd.read_csv(test_path,  index_col=0, parse_dates=True)

    return train_df, val_df, test_df


# ------------------------------------------------------------------ #
# Dynamic multi-ticker pipeline (used by the Streamlit dashboard)
# ------------------------------------------------------------------ #

def fetch_and_process(ticker: str,
                      start: str = "2013-01-01",
                      end: str   = "2024-12-31"):
    """
    Full pipeline for any ticker — download, feature-engineer, scale, split.

    The feature set and date splits exactly match the AAPL training setup
    so the existing models can score any ticker's observations.

    Parameters
    ----------
    ticker : str  e.g. "MSFT", "TSLA"
    start  : str  earliest date to download (YYYY-MM-DD)
    end    : str  latest  date to download  (YYYY-MM-DD)

    Returns
    -------
    train_df : pd.DataFrame  — scaled features, train period (up to 2019-12-31)
    val_df   : pd.DataFrame  — scaled features, val period   (2020–2021)
    test_df  : pd.DataFrame  — scaled features, test period  (2022–2024)
    raw_df   : pd.DataFrame  — raw OHLCV for the full period (un-scaled)
    """
    # 1. Download raw OHLCV (uses on-disk cache in data/raw/)
    raw = download(ticker, start=start, end=end)

    # 2. Engineer features
    featured = engineer_features(raw)

    # 3. Split by date (same boundaries as AAPL training)
    train_feat = featured[featured.index <= TRAIN_END]
    val_feat   = featured[(featured.index >= VAL_START) & (featured.index <= VAL_END)]
    test_feat  = featured[featured.index >= TEST_START]

    if len(train_feat) == 0:
        raise ValueError(
            f"No training-period data found for {ticker} before {TRAIN_END}. "
            "This can happen if the ticker was listed after 2013, the download "
            "timed out, or the symbol is invalid. Please try again."
        )

    # 4. Scale — fit ONLY on train, transform all splits
    scaler    = RobustScaler()
    train_arr = scaler.fit_transform(train_feat.values)
    val_arr   = scaler.transform(val_feat.values)   if len(val_feat)  > 0 else val_feat.values
    test_arr  = scaler.transform(test_feat.values)  if len(test_feat) > 0 else test_feat.values

    cols = featured.columns.tolist()

    train_df = pd.DataFrame(train_arr, index=train_feat.index, columns=cols)
    val_df   = pd.DataFrame(val_arr,   index=val_feat.index,   columns=cols)
    test_df  = pd.DataFrame(test_arr,  index=test_feat.index,  columns=cols)

    # 5. Raw sub-splits aligned to the feature index (needed by TradingEnv)
    raw_df = raw.loc[featured.index]

    return train_df, val_df, test_df, raw_df


# ------------------------------------------------------------------ #
# Live observation builder  (used by live/paper_trader.py)
# ------------------------------------------------------------------ #

def fetch_live_obs(ticker: str,
                   window_size: int = 10,
                   warmup_start: str = "2013-01-01"):
    """
    Fetch the latest daily bar for `ticker` and return a model-ready
    observation window, scaled consistently with the training pipeline.

    Steps:
        1. Download OHLCV from warmup_start through today (disk-cached).
           Force-refresh the cache if the last bar is more than 2 days old.
        2. Engineer the same 8 features as training
           (Close, RSI, MACD, EMA20, EMA50, BBWidth, OBV, ATR).
        3. Fit RobustScaler on the training-period rows only (pre-2020)
           then transform the full history — identical to fetch_and_process.
        4. Return the last window_size scaled rows as the observation array.

    Returns
    -------
    obs           : np.ndarray shape (window_size, 8), dtype float32
    current_price : float   most recent raw Close price (USD)
    latest_date   : pd.Timestamp  date of the most recent bar
    featured      : pd.DataFrame  full feature history (for diagnostics)
    """
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    # Refresh stale cache (older than 2 calendar days)
    cache_path = RAW_DIR / f"{ticker}.csv"
    if cache_path.exists():
        try:
            cached     = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            days_stale = (pd.Timestamp.today() - cached.index[-1]).days
            if days_stale > 2:
                cache_path.unlink()
        except Exception:
            cache_path.unlink()

    raw      = download(ticker, start=warmup_start, end=today)
    featured = engineer_features(raw)

    if len(featured) < window_size:
        raise ValueError(
            f"Not enough rows for {ticker}: got {len(featured)}, need >= {window_size}."
        )

    train_mask = featured.index <= TRAIN_END
    if train_mask.sum() < 50:
        raise ValueError(
            f"Fewer than 50 pre-2020 rows for {ticker}. Ticker may be too new."
        )

    scaler    = RobustScaler()
    scaler.fit(featured[train_mask].values)
    scaled_arr = scaler.transform(featured.values)
    scaled     = pd.DataFrame(scaled_arr, index=featured.index, columns=featured.columns)

    obs           = scaled.iloc[-window_size:].values.astype(np.float32)
    current_price = float(raw["Close"].iloc[-1])
    latest_date   = featured.index[-1]

    return obs, current_price, latest_date, featured