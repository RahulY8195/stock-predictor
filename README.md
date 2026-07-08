# Stock Predictor

A dashboard for exploring historical stock data, technical indicators, and short-horizon ML price forecasts, with every forecast logged and reconciled against actual outcomes.

Forecast accuracy is typically reported as an in-sample fit metric — how well a model reproduces the data it was trained on. This project tracks a second, independent number: every prediction is persisted with its target date, and a reconciliation job checks it against the real closing price once that date passes. Reported accuracy is out-of-sample.

## Architecture

- **FastAPI backend** (`app/`) — owns model training, forecasting, and persistence. Streamlit is a client of this API.
- **PostgreSQL** — every forecasted `(ticker, target_date, predicted_price)` is stored as a row; a reconciliation job fills in the real closing price once the target date passes.
- **APScheduler** — runs reconciliation daily in the background. A `POST /reconcile` endpoint (and a UI button) triggers it manually.
- **Streamlit UI** — sidebar controls for ticker/model/window/horizon, an interactive Plotly chart, and a tracked-accuracy panel showing out-of-sample MAE/RMSE/MAPE alongside in-sample fit metrics.

```
app/
  database.py    SQLAlchemy engine/session (PostgreSQL)
  models.py      Prediction ORM model
  schemas.py     Pydantic request/response models
  indicators.py  RSI / MACD / moving averages
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
- In-sample fit metrics (MAE, RMSE, MAPE) on historical data
- Out-of-sample accuracy tracking: forecasts are scored against reality once their target date passes
- Interactive Plotly chart with actual price, moving averages, model fit, and forecast band
- REST API (`/predict`, `/stock/{ticker}`, `/accuracy/{ticker}`, `/predictions/{ticker}`, `/reconcile`) — docs at `/docs`

## Tech Stack

Python, FastAPI, SQLAlchemy + PostgreSQL, APScheduler, scikit-learn, yfinance, Streamlit, Plotly, pytest, Docker Compose, GitHub Actions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Start PostgreSQL:

```bash
docker compose up -d db
```

## Run

```bash
# terminal 1
uvicorn app.main:app --reload

# terminal 2
streamlit run app.py
```

UI: http://localhost:8501. API docs: http://localhost:8000/docs.

## Tests

```bash
pytest -v
```

All network calls (`yfinance`) are mocked and the test suite runs against an isolated in-memory SQLite database, independent of the PostgreSQL container. No external services required.

## How accuracy tracking works

1. `POST /predict` trains a model, generates a forecast, and logs each `(target_date, predicted_price)` pair.
2. A background job (or manual `POST /reconcile`) checks every prediction whose `target_date` has passed and fills in the real closing price.
3. `GET /accuracy/{ticker}` aggregates MAE/RMSE/MAPE over resolved predictions only — the out-of-sample number, distinct from the in-sample fit metrics reported at prediction time.
4. The dashboard displays both, labeled separately.
