from datetime import date

from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.data import fetch_close_on_or_after
from app.models import Prediction


def log_predictions(
    db: Session,
    ticker: str,
    model_name: str,
    window: int,
    made_on: date,
    forecast: list[tuple[date, float]],
) -> list[Prediction]:
    rows = [
        Prediction(
            ticker=ticker,
            model_name=model_name,
            window=window,
            made_on=made_on,
            target_date=target_date,
            predicted_price=predicted_price,
        )
        for target_date, predicted_price in forecast
    ]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


def reconcile_actuals(db: Session, fetch_close_fn=None) -> dict:
    if fetch_close_fn is None:
        fetch_close_fn = fetch_close_on_or_after

    today = date.today()
    pending = (
        db.query(Prediction)
        .filter(and_(Prediction.actual_price.is_(None), Prediction.target_date <= today))
        .all()
    )

    resolved = 0
    for prediction in pending:
        actual = fetch_close_fn(prediction.ticker, prediction.target_date)
        if actual is not None:
            prediction.actual_price = actual
            resolved += 1

    db.commit()
    return {"checked": len(pending), "resolved": resolved}
