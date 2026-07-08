from datetime import date, timedelta

import pandas as pd
import pytest

from app.backtest import log_predictions


@pytest.fixture(autouse=True)
def mock_market_data(monkeypatch, synthetic_price_df):
    monkeypatch.setattr("app.main.fetch_data", lambda ticker, period: synthetic_price_df)
    monkeypatch.setattr(
        "app.main.fetch_info",
        lambda ticker: {"name": f"{ticker} Inc.", "logo_b64": ""},
    )
    monkeypatch.setattr("app.backtest.fetch_close_on_or_after", lambda ticker, on_date: 123.45)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_info_endpoint(client):
    resp = client.get("/info/aapl")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["name"] == "AAPL Inc."


def test_stock_endpoint_returns_indicators(client):
    resp = client.get("/stock/AAPL", params={"period": "2y"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert len(body["history"]) > 0
    assert "RSI" in body["history"][-1]
    assert "MACD" in body["history"][-1]


def test_stock_endpoint_404_when_no_data(client, monkeypatch):
    monkeypatch.setattr("app.main.fetch_data", lambda ticker, period: pd.DataFrame())
    resp = client.get("/stock/BADTICKER")
    assert resp.status_code == 404


def test_predict_returns_forecast_and_logs_predictions(client):
    resp = client.post(
        "/predict",
        json={
            "ticker": "aapl",
            "period": "2y",
            "model": "Gradient Boosting",
            "window": 30,
            "forecast_days": 10,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert len(body["forecast"]) == 10
    assert body["fit_mae"] >= 0

    log_resp = client.get("/predictions/AAPL")
    assert log_resp.status_code == 200
    assert len(log_resp.json()) == 10


def test_predict_400_when_window_larger_than_history(client, monkeypatch, synthetic_price_df):
    short_df = synthetic_price_df.tail(20)
    monkeypatch.setattr("app.main.fetch_data", lambda ticker, period: short_df)

    resp = client.post(
        "/predict",
        json={"ticker": "AAPL", "window": 60, "forecast_days": 10},
    )
    assert resp.status_code == 400


def test_accuracy_is_empty_before_any_predictions_resolve(client):
    resp = client.get("/accuracy/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved_predictions"] == 0
    assert body["mae"] is None


def test_predict_targets_are_all_future_so_nothing_resolves_yet(client):
    """A freshly logged forecast targets future trading days, so reconcile
    shouldn't touch it until those dates actually pass."""
    client.post(
        "/predict",
        json={"ticker": "AAPL", "window": 30, "forecast_days": 5},
    )

    reconcile_resp = client.post("/reconcile")
    assert reconcile_resp.status_code == 200
    assert reconcile_resp.json()["resolved"] == 0

    accuracy_resp = client.get("/accuracy/AAPL")
    assert accuracy_resp.json()["resolved_predictions"] == 0


def test_reconcile_then_accuracy_reflects_resolved_predictions(client, db_session):
    """Predictions with a target_date in the past should resolve and feed accuracy."""
    db = db_session()
    log_predictions(
        db, "AAPL", "Gradient Boosting", 30, date.today() - timedelta(days=5),
        [
            (date.today() - timedelta(days=2), 100.0),
            (date.today() - timedelta(days=1), 110.0),
        ],
    )
    db.close()

    reconcile_resp = client.post("/reconcile")
    assert reconcile_resp.status_code == 200
    assert reconcile_resp.json()["resolved"] == 2

    accuracy_resp = client.get("/accuracy/AAPL")
    body = accuracy_resp.json()
    assert body["resolved_predictions"] == 2
    assert body["mae"] is not None
