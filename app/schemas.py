from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    ticker: str
    period: Literal["1y", "2y", "3y", "5y"] = "2y"
    model: Literal["Gradient Boosting", "MLP Neural Network"] = "Gradient Boosting"
    window: int = 60
    forecast_days: int = 30


class ForecastPoint(BaseModel):
    date: date
    predicted_price: float


class PredictResponse(BaseModel):
    ticker: str
    model: str
    window: int
    latest_close: float
    fit_mae: float
    fit_rmse: float
    fit_mape: float
    forecast: list[ForecastPoint]


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticker: str
    model_name: str
    window: int
    made_on: date
    target_date: date
    predicted_price: float
    actual_price: float | None
    created_at: datetime


class AccuracyResponse(BaseModel):
    ticker: str
    resolved_predictions: int
    mae: float | None
    rmse: float | None
    mape: float | None


class ReconcileResponse(BaseModel):
    checked: int
    resolved: int
