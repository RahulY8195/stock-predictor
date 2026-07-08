from datetime import date, timedelta

from app.backtest import log_predictions, reconcile_actuals
from app.models import Prediction


def test_log_predictions_inserts_rows(db_session):
    db = db_session()
    forecast = [
        (date.today() + timedelta(days=1), 101.5),
        (date.today() + timedelta(days=2), 102.25),
    ]

    rows = log_predictions(db, "AAPL", "Gradient Boosting", 60, date.today(), forecast)

    assert len(rows) == 2
    assert all(r.id is not None for r in rows)
    assert db.query(Prediction).count() == 2
    db.close()


def test_reconcile_actuals_resolves_past_predictions_only(db_session, monkeypatch):
    db = db_session()
    today = date.today()

    log_predictions(
        db, "AAPL", "Gradient Boosting", 60, today - timedelta(days=5),
        [(today - timedelta(days=2), 100.0)],  # past, unresolved -> should resolve
    )
    log_predictions(
        db, "AAPL", "Gradient Boosting", 60, today,
        [(today + timedelta(days=5), 200.0)],  # future -> should NOT be touched
    )

    monkeypatch.setattr("app.backtest.fetch_close_on_or_after", lambda ticker, on_date: 105.0)

    result = reconcile_actuals(db)

    assert result == {"checked": 1, "resolved": 1}

    resolved = db.query(Prediction).filter(Prediction.predicted_price == 100.0).one()
    assert resolved.actual_price == 105.0

    untouched = db.query(Prediction).filter(Prediction.predicted_price == 200.0).one()
    assert untouched.actual_price is None
    db.close()


def test_reconcile_actuals_leaves_prediction_unresolved_if_no_market_data(db_session, monkeypatch):
    db = db_session()
    today = date.today()

    log_predictions(
        db, "AAPL", "Gradient Boosting", 60, today - timedelta(days=5),
        [(today - timedelta(days=2), 100.0)],
    )

    monkeypatch.setattr("app.backtest.fetch_close_on_or_after", lambda ticker, on_date: None)

    result = reconcile_actuals(db)

    assert result == {"checked": 1, "resolved": 0}
    row = db.query(Prediction).one()
    assert row.actual_price is None
    db.close()
