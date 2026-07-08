# Stock Predictor

An interactive dashboard for exploring historical stock data, technical indicators, and short-horizon ML price forecasts — with every forecast logged and checked against what actually happened.

Most "ML stock predictor" projects only report how well a model fits data it already trained on, which is a meaningless accuracy claim. This one is architected differently: every prediction is persisted with its target date, and a scheduled job reconciles it against the real closing price once that date arrives, so the accuracy shown is genuinely out-of-sample.

## Architecture

- **FastAPI backend** (`app/`) — trains models, generates forecasts, and owns all persistence. Streamlit is a thin client of this API, not where the logic lives.
- **SQLite** — every forecasted `(ticker, target_date, predicted_price)` is logged as a row; once the target date passes, a reconciliation job fills in the real closing price.
- **APScheduler** — runs the reconciliation job daily in the background while the API is up. There's also a `/reconcile` endpoint you can hit manually (the UI has a button for this).
- **Streamlit UI** — sidebar controls for ticker/model/window/horizon, an interactive Plotly chart, and a "Tracked Accuracy" panel showing real out-of-sample MAE/RMSE/MAPE, separate from the in-sample fit metrics.

```
app/
  database.py    SQLAlchemy engine/session (SQLite)
  models.py      Prediction ORM model
  schemas.py     Pydantic request/response models
  indicators.py  RSI / MACD / moving averages (pure functions)
  ml.py          model training, forecasting, fit metrics
  data.py        yfinance wrappers
  backtest.py    prediction logging + reconciliation
  main.py        FastAPI app and routes
app.py           Streamlit UI (API client)
tests/           pytest suite, network calls mocked
```

## Features

- Live price data via `yfinance` for any ticker
- Technical indicators: 20/50-day moving averages, RSI, MACD
- Forecasting with a choice of two models: Gradient Boosting or an MLP neural network
- In-sample fit metrics (MAE, RMSE, MAPE) computed on historical data
- **Out-of-sample accuracy tracking**: every forecast is logged and scored against reality once its target date passes
- Interactive Plotly chart with actual price, moving averages, model fit, and forecast band
- REST API (`/predict`, `/stock/{ticker}`, `/accuracy/{ticker}`, `/predictions/{ticker}`, `/reconcile`) — full docs at `/docs`

## Tech Stack

Python, FastAPI, SQLAlchemy + SQLite, APScheduler, scikit-learn, yfinance, Streamlit, Plotly, pytest, GitHub Actions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the API first, then the UI in a separate terminal:

```bash
# terminal 1
uvicorn app.main:app --reload

# terminal 2
streamlit run app.py
```

Open http://localhost:8501 for the UI, or http://localhost:8000/docs for the API.

## Tests

```bash
pytest -v
```

Tests mock all network calls (`yfinance`) and use an in-memory SQLite database, so they run without any external services or API keys.

## How the accuracy tracking works

1. `POST /predict` trains a model, generates a forecast, and logs each `(target_date, predicted_price)` pair to the database.
2. A background job (or a manual `POST /reconcile`) checks every logged prediction whose `target_date` has passed and fills in the real closing price from `yfinance`.
3. `GET /accuracy/{ticker}` aggregates MAE/RMSE/MAPE only over *resolved* predictions — this is the number that actually means something, as opposed to in-sample fit metrics which just show how well a model memorized its own training data.
4. The dashboard shows both, clearly labeled, so the distinction is impossible to miss.

## Disclaimer

This is a technical/engineering demo, not investment advice. Short-horizon price forecasting from technical indicators has weak predictive power in practice — the point of this project is the accuracy-tracking architecture, not a claim that the forecasts are reliable.
