"""
live/paper_trader.py
--------------------
Phase 1: EOD paper trading via Alpaca Markets.

What it does (once per trading day at ~4:05 PM ET)
---------------------------------------------------
1.  Fetch the latest daily OHLCV for the configured ticker via yfinance.
2.  Run the same 8-feature engineering + RobustScaler pipeline used during
    training to build a 10-day observation window.
3.  Load the trained PPO model and call model.predict(obs).
4.  Translate the action (0=Hold / 1=Buy 10% / 2=Sell all) into an Alpaca
    paper order.
5.  Append a structured row to live/trade_log.csv for review.

Setup
-----
1.  Create a free Alpaca paper account at https://app.alpaca.markets
2.  Generate a paper API key pair (Paper Trading → API Keys)
3.  cp live/.env.example live/.env  and fill in your keys
4.  Activate the project venv:  source .venv/bin/activate
5.  Run once now to test:       python live/paper_trader.py --now
6.  Run the daily daemon:       python live/paper_trader.py

Flags
-----
--now      Execute the trading logic immediately and exit (good for testing)
--ticker   Override the ticker in .env  (e.g. --ticker MSFT)
--model    Override the model in .env   (e.g. --model ppo_aapl)
--dry-run  Compute decision but do NOT place any order (safe to run anytime)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------ #
# Path setup — make project root importable before any local imports  #
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------ #
# Load .env credentials                                               #
# ------------------------------------------------------------------ #
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import schedule
from stable_baselines3 import PPO, DQN
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from data.pipeline import fetch_live_obs

# ------------------------------------------------------------------ #
# Logging                                                             #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "live" / "trader.log", mode="a"),
    ],
)
log = logging.getLogger("paper_trader")

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #
ACTION_NAMES   = {0: "HOLD", 1: "BUY", 2: "SELL"}
WINDOW_SIZE    = 10         # must match model training (env window_size=10)
BUY_FRACTION   = 0.10      # invest 10% of cash per buy (matches TradingEnv)
TRADE_LOG_PATH = PROJECT_ROOT / "live" / "trade_log.csv"
TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Helper utilities                                                    #
# ------------------------------------------------------------------ #

def _load_model(model_name: str):
    """Load a trained SB3 model from models/. Returns (model, algo_name)."""
    path = PROJECT_ROOT / "models" / f"{model_name}.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Available models in models/: "
            + ", ".join(p.stem for p in (PROJECT_ROOT / "models").glob("*.zip"))
        )
    # DQN models are named dqn_*; everything else is PPO
    if model_name.startswith("dqn"):
        return DQN.load(path), "DQN"
    return PPO.load(path), "PPO"


def _get_equity(client: TradingClient) -> float:
    """Return current paper account equity in USD."""
    acct = client.get_account()
    return float(acct.equity)


def _get_cash(client: TradingClient) -> float:
    """Return current paper account cash in USD."""
    acct = client.get_account()
    return float(acct.cash)


def _get_position_qty(client: TradingClient, ticker: str) -> float:
    """Return current share count for ticker, or 0 if not held."""
    try:
        pos = client.get_open_position(ticker)
        return float(pos.qty)
    except Exception:
        return 0.0


def _log_trade(row: dict):
    """Append a trade record to the CSV log."""
    file_exists = TRADE_LOG_PATH.exists()
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _is_market_open(client: TradingClient) -> bool:
    """Check if the US equity market is currently open."""
    clock = client.get_clock()
    return clock.is_open


# ------------------------------------------------------------------ #
# Core trading logic                                                  #
# ------------------------------------------------------------------ #

def run_trading_cycle(
    ticker:   str,
    model_name: str,
    dry_run:  bool,
    client:   TradingClient,
    model,
    algo_name: str,
):
    """
    Execute one complete trading cycle:
        fetch obs → predict → place order → log
    """
    log.info("=" * 60)
    log.info(f"Trading cycle | ticker={ticker} | model={model_name} | dry_run={dry_run}")
    log.info("=" * 60)

    # ----------------------------------------------------------------
    # 1. Fetch latest observation
    # ----------------------------------------------------------------
    log.info(f"Fetching latest daily data for {ticker}...")
    obs, current_price, latest_date, _ = fetch_live_obs(ticker, window_size=WINDOW_SIZE)

    log.info(f"  Latest bar : {latest_date.date()}")
    log.info(f"  Close price: ${current_price:,.2f}")

    # ----------------------------------------------------------------
    # 2. Model inference
    # ----------------------------------------------------------------
    action, _ = model.predict(obs, deterministic=True)
    action     = int(action)
    action_str = ACTION_NAMES[action]
    log.info(f"  Model decision: {action_str} (action={action})")

    # ----------------------------------------------------------------
    # 3. Get current account state
    # ----------------------------------------------------------------
    equity    = _get_equity(client)
    cash      = _get_cash(client)
    held_qty  = _get_position_qty(client, ticker)
    held_value = held_qty * current_price

    log.info(f"  Account equity : ${equity:,.2f}")
    log.info(f"  Cash           : ${cash:,.2f}")
    log.info(f"  {ticker} held  : {held_qty:.4f} shares (${held_value:,.2f})")

    # ----------------------------------------------------------------
    # 4. Execute order
    # ----------------------------------------------------------------
    order_placed   = False
    shares_traded  = 0.0
    order_notional = 0.0

    if action == 0:   # HOLD
        log.info("  Action: HOLD — no order placed.")

    elif action == 1:  # BUY — spend BUY_FRACTION of available cash
        spend = cash * BUY_FRACTION
        if spend < 1.0:
            log.warning("  BUY skipped — insufficient cash (< $1.00 available).")
        else:
            log.info(f"  BUY ${spend:,.2f} notional of {ticker}")
            if not dry_run:
                req = MarketOrderRequest(
                    symbol         = ticker,
                    notional       = round(spend, 2),   # fractional shares via notional
                    side           = OrderSide.BUY,
                    time_in_force  = TimeInForce.DAY,
                )
                client.submit_order(req)
                order_placed   = True
                order_notional = spend
                shares_traded  = spend / current_price
                log.info(f"  ✓ BUY order submitted: ${spend:,.2f} (~{shares_traded:.4f} shares)")
            else:
                log.info("  [DRY RUN] BUY order NOT submitted.")

    elif action == 2:  # SELL — liquidate entire position
        if held_qty <= 0:
            log.info("  SELL skipped — no shares held.")
        else:
            log.info(f"  SELL all {held_qty:.4f} shares of {ticker} (${held_value:,.2f})")
            if not dry_run:
                req = MarketOrderRequest(
                    symbol        = ticker,
                    qty           = held_qty,
                    side          = OrderSide.SELL,
                    time_in_force = TimeInForce.DAY,
                )
                client.submit_order(req)
                order_placed   = True
                order_notional = held_value
                shares_traded  = held_qty
                log.info(f"  ✓ SELL order submitted: {held_qty:.4f} shares")
            else:
                log.info("  [DRY RUN] SELL order NOT submitted.")

    # ----------------------------------------------------------------
    # 5. Log trade record
    # ----------------------------------------------------------------
    record = {
        "timestamp"      : datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "bar_date"       : str(latest_date.date()),
        "ticker"         : ticker,
        "model"          : model_name,
        "algo"           : algo_name,
        "action"         : action_str,
        "close_price"    : round(current_price, 4),
        "notional_usd"   : round(order_notional, 2),
        "shares_traded"  : round(shares_traded, 6),
        "equity_after"   : round(equity, 2),
        "cash_after"     : round(cash, 2),
        "held_qty_before": round(held_qty, 6),
        "order_placed"   : order_placed,
        "dry_run"        : dry_run,
    }
    _log_trade(record)
    log.info(f"  Trade record written → {TRADE_LOG_PATH.name}")
    log.info("=" * 60)


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 EOD paper trader — runs the RL agent once per trading day."
    )
    parser.add_argument("--now",     action="store_true",
                        help="Run the trading cycle immediately and exit.")
    parser.add_argument("--ticker",  default=None,
                        help="Override TRADE_TICKER from .env")
    parser.add_argument("--model",   default=None,
                        help="Override MODEL_NAME from .env (e.g. ppo_aapl, ppo_multi)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute the decision but do NOT submit any order.")
    args = parser.parse_args()

    # -- Configuration --------------------------------------------------
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    ticker     = args.ticker or os.getenv("TRADE_TICKER",  "AAPL")
    model_name = args.model  or os.getenv("MODEL_NAME",    "ppo_multi")
    dry_run    = args.dry_run

    if not api_key or api_key == "your_paper_api_key_here":
        log.error(
            "ALPACA_API_KEY not set.\n"
            "  1. Create a free paper account at https://app.alpaca.markets\n"
            "  2. Copy live/.env.example → live/.env and fill in your keys\n"
            "  3. Re-run this script."
        )
        sys.exit(1)

    # -- Initialise Alpaca client (paper=True) --------------------------
    log.info(f"Connecting to Alpaca paper trading account...")
    client = TradingClient(api_key, secret_key, paper=True)
    acct   = client.get_account()
    log.info(f"  Account ID : {acct.id}")
    log.info(f"  Equity     : ${float(acct.equity):,.2f}")
    log.info(f"  Cash       : ${float(acct.cash):,.2f}")

    # -- Load model -----------------------------------------------------
    log.info(f"Loading model: {model_name}")
    model, algo_name = _load_model(model_name)
    log.info(f"  Model loaded: {algo_name} ({model_name})")

    # -- Build the trading job ------------------------------------------
    def job():
        try:
            # Skip if market is closed (weekends, holidays)
            if not _is_market_open(client):
                log.info("Market is closed today — skipping cycle.")
                return
            run_trading_cycle(ticker, model_name, dry_run, client, model, algo_name)
        except Exception as e:
            log.exception(f"Trading cycle failed: {e}")

    if args.now:
        # --now: run immediately (ignores market-closed check for testing)
        log.info("--now flag set: running trading cycle immediately.")
        try:
            run_trading_cycle(ticker, model_name, dry_run, client, model, algo_name)
        except Exception as e:
            log.exception(f"Trading cycle failed: {e}")
        log.info("Done. Exiting.")
        sys.exit(0)

    # -- Daemon mode: run at 16:05 ET daily -----------------------------
    # (US markets close at 16:00 ET; 5-min buffer for EOD bar formation)
    RUN_TIME_ET = "16:05"
    log.info(
        f"Daemon mode: scheduled to run at {RUN_TIME_ET} ET every trading day.\n"
        f"  Ticker : {ticker}\n"
        f"  Model  : {model_name}\n"
        f"  Dry run: {dry_run}\n"
        "Press Ctrl+C to stop."
    )
    schedule.every().day.at(RUN_TIME_ET).do(job)

    # Also run immediately on startup so you don't wait until tomorrow
    log.info("Running once on startup (will skip if market is closed)...")
    job()

    while True:
        schedule.run_pending()
        time.sleep(30)   # check every 30 seconds


if __name__ == "__main__":
    main()
