import logging
from contextlib import asynccontextmanager
from datetime import date

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.backtest import log_predictions, reconcile_actuals
from app.data import fetch_data, fetch_info
from app.database import SessionLocal, get_db, init_db
from app.indicators import add_indicators
from app.ml import fit_metrics, predict_future, train_model
from app.models import Prediction
from app.schemas import (
    AccuracyResponse,
    ForecastPoint,
    PredictRequest,
    PredictResponse,
    PredictionOut,
    ReconcileResponse,
)

logger = logging.getLogger("stock_predictor")

scheduler = BackgroundScheduler()


def _scheduled_reconcile():
    db = SessionLocal()
    try:
        result = reconcile_actuals(db)
        logger.info("Scheduled reconcile: %s", result)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler.add_job(_scheduled_reconcile, "interval", hours=24, id="daily_reconcile")
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="Stock Predictor API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info/{ticker}")
def get_info(ticker: str):
    ticker = ticker.upper().strip()
    info = fetch_info(ticker)
    return {"ticker": ticker, "name": info["name"], "logo_b64": info["logo_b64"]}


@app.get("/stock/{ticker}")
def get_stock(ticker: str, period: str = "2y"):
    ticker = ticker.upper().strip()
    df = fetch_data(ticker, period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    df = add_indicators(df)
    info = fetch_info(ticker)

    records = df.reset_index()
    records["Date"] = records["Date"].dt.strftime("%Y-%m-%d")
    columns = ["Date", "Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD", "Signal"]
    history = records[columns].replace({np.nan: None}).to_dict(orient="records")

    return {"ticker": ticker, "name": info["name"], "logo_b64": info["logo_b64"], "history": history}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    ticker = req.ticker.upper().strip()
    df = fetch_data(ticker, req.period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    if len(df) <= req.window:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough history ({len(df)} rows) for a {req.window}-day window",
        )

    close_series = df["Close"].squeeze().values.astype(float)
    model, scaler = train_model(close_series, req.window, req.model)
    metrics = fit_metrics(model, scaler, close_series, req.window)
    future_prices = predict_future(model, scaler, close_series[-req.window :], req.forecast_days)

    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=req.forecast_days + 1)[1:]

    forecast = [
        (fd.date(), float(fp)) for fd, fp in zip(future_dates, future_prices)
    ]
    log_predictions(db, ticker, req.model, req.window, date.today(), forecast)

    return PredictResponse(
        ticker=ticker,
        model=req.model,
        window=req.window,
        latest_close=float(close_series[-1]),
        fit_mae=metrics["mae"],
        fit_rmse=metrics["rmse"],
        fit_mape=metrics["mape"],
        forecast=[ForecastPoint(date=d, predicted_price=p) for d, p in forecast],
    )


@app.get("/predictions/{ticker}", response_model=list[PredictionOut])
def get_predictions(ticker: str, db: Session = Depends(get_db)):
    ticker = ticker.upper().strip()
    rows = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker)
        .order_by(Prediction.created_at.desc())
        .limit(500)
        .all()
    )
    return rows


@app.get("/accuracy/{ticker}", response_model=AccuracyResponse)
def get_accuracy(ticker: str, db: Session = Depends(get_db)):
    ticker = ticker.upper().strip()
    rows = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker, Prediction.actual_price.isnot(None))
        .all()
    )

    if not rows:
        return AccuracyResponse(ticker=ticker, resolved_predictions=0, mae=None, rmse=None, mape=None)

    errors = np.array([r.predicted_price - r.actual_price for r in rows])
    actuals = np.array([r.actual_price for r in rows])
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(np.abs(errors / actuals)) * 100)

    return AccuracyResponse(ticker=ticker, resolved_predictions=len(rows), mae=mae, rmse=rmse, mape=mape)


@app.post("/reconcile", response_model=ReconcileResponse)
def reconcile(db: Session = Depends(get_db)):
    result = reconcile_actuals(db)
    return ReconcileResponse(**result)
