import yfinance as yf, os
import pandas as pd

def download(ticker="AAPL", start="2013-01-01", end="2024-12-31"):
    path = f"data/raw/{ticker}.csv"
    if not os.path.exists(path):
        df = yf.download(ticker, start=start, end=end)
        df.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)