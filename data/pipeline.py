import os
from pathlib import Path

import yfinance as yf
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def download(ticker="AAPL", start="2013-01-01", end="2024-12-31"):
    """
    Download raw OHLCV (or load from disk if already present).

    IMPORTANT: Paths are resolved relative to the project root, not the
    current working directory (so notebooks work even when `cwd=notebooks/`).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.csv"

    if not path.exists():
        df = yf.download(ticker, start=start, end=end)
        df.to_csv(path)

    return pd.read_csv(path, index_col=0, parse_dates=True)


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
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

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
    val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

    return train_df, val_df, test_df