import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session):
    return TestClient(app)


@pytest.fixture()
def synthetic_price_df():
    """A deterministic, gently-trending price series with enough rows to train on."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    trend = np.linspace(100, 140, n)
    noise = rng.normal(0, 0.8, n)
    close = trend + noise

    df = pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 0.6,
            "Low": close - 0.6,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df
