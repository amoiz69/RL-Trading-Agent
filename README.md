# RL Trading Agent

An end-to-end reinforcement learning system that trains DQN and PPO agents to trade AAPL stock. Built as a portfolio project demonstrating the full ML lifecycle — data pipeline, custom environment, agent training, backtesting, and live deployment.

## Live demo
https://rl-trading-agent-kweeynevx86uzry4stwcjx.streamlit.app

## Results (test set 2022–2024)

| Metric | DQN | PPO | Buy & Hold |
|---|---|---|---|
| Total return | 38.5% | 11.01 | 40.83 |
| Sharpe ratio | 0.56 | 0.276 | 0.559 |
| Max drawdown | -31.2 | -29.11 | -30.91 |
| Calmar ratio | 0.375 | 0.124 | 0.393 |
| Trades | 657 | 263 | 1 |



## Project structure

```
rl-trading-agent/
├── data/
│   ├── raw/              # downloaded OHLCV CSVs (gitignored)
│   └── processed/        # normalised train/val/test splits (gitignored)
├── env/
│   └── trading_env.py    # custom Gymnasium environment
├── agents/
│   ├── dqn_agent.py      # DQN training script
│   └── ppo_agent.py      # PPO training script
├── backtest/
│   └── backtester.py     # evaluation metrics + QuantStats tearsheet
├── dashboard/
│   └── app.py            # Streamlit web app
├── models/               # saved .zip model weights
├── reports/              # backtest charts + HTML tearsheets
├── notebooks/            # phase-by-phase development notebooks
├── train.py              # top-level training orchestrator
└── requirements.txt
```

## How to run locally

```bash
git clone https://github.com/your-username/rl-trading-agent
cd rl-trading-agent
pip install -r requirements.txt

# Run the full data pipeline
python data/pipeline.py

# Train both agents
python train.py

# Run backtesting
python backtest/backtester.py

# Launch the dashboard
streamlit run dashboard/app.py
```

## Architecture

**MDP formulation**
- State: sliding window of 10 days × 8 features (Close, RSI, MACD, EMA20, EMA50, BB width, OBV, ATR) — normalised with RobustScaler
- Action: Discrete(3) — Hold / Buy (10% of cash) / Sell (all shares)
- Reward: percentage change in portfolio value per step

**Agents**
- DQN: value-based, replay buffer size 50k, epsilon-greedy exploration
- PPO: policy-based, clipped surrogate objective, GAE advantage estimation

**Data splits** (chronological — no leakage)
- Train: 2013–2019 (bull market)
- Validation: 2020–2021 (COVID crash + recovery)
- Test: 2022–2024 (rate hike bear market + rebound)

## Tech stack

`Python` `PyTorch` `Stable-Baselines3` `Gymnasium` `Pandas` `scikit-learn` `ta` `QuantStats` `Plotly` `Streamlit`

## Phases

| Phase | What was built |
|---|---|
| 1 | Project setup, folder structure, dependencies |
| 2 | Data pipeline — yfinance download, 6 technical indicators, RobustScaler, chronological split |
| 3 | Custom Gymnasium environment — MDP design, transaction costs, trade logging |
| 4 | Agent training — DQN + PPO with EvalCallback, W&B experiment tracking |
| 5 | Backtesting — 6 risk-adjusted metrics, QuantStats tearsheet, trade signal charts |
| 6 | Streamlit dashboard — live agent inference, deployed to Streamlit Cloud |