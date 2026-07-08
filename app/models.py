from datetime import datetime, timezone

from sqlalchemy import Column, Date, DateTime, Float, Integer, String

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    window = Column(Integer, nullable=False)
    made_on = Column(Date, nullable=False)
    target_date = Column(Date, index=True, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
