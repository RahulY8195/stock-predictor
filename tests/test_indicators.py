import numpy as np
import pandas as pd

from app.indicators import add_indicators


def _df_from_closes(closes):
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=dates)


def test_moving_averages_match_pandas_rolling():
    closes = np.linspace(100, 200, 80)
    df = add_indicators(_df_from_closes(closes))

    expected_ma20 = pd.Series(closes).rolling(20).mean().values
    expected_ma50 = pd.Series(closes).rolling(50).mean().values

    np.testing.assert_allclose(df["MA20"].values, expected_ma20, equal_nan=True)
    np.testing.assert_allclose(df["MA50"].values, expected_ma50, equal_nan=True)


def test_rsi_is_100_for_strictly_increasing_series():
    closes = np.arange(100, 160, 1.0)  # every day is a gain, no losses
    df = add_indicators(_df_from_closes(closes))

    rsi_tail = df["RSI"].iloc[20:]
    assert (rsi_tail == 100).all()


def test_rsi_is_0_for_strictly_decreasing_series():
    closes = np.arange(160, 100, -1.0)  # every day is a loss, no gains
    df = add_indicators(_df_from_closes(closes))

    rsi_tail = df["RSI"].iloc[20:]
    assert (rsi_tail == 0).all()


def test_macd_positive_in_sustained_uptrend():
    closes = np.linspace(100, 250, 100)
    df = add_indicators(_df_from_closes(closes))

    assert df["MACD"].iloc[-1] > 0
    assert df["MACD"].iloc[-1] > df["Signal"].iloc[-1]


def test_macd_negative_in_sustained_downtrend():
    closes = np.linspace(250, 100, 100)
    df = add_indicators(_df_from_closes(closes))

    assert df["MACD"].iloc[-1] < 0
    assert df["MACD"].iloc[-1] < df["Signal"].iloc[-1]
