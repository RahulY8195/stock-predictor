import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    df = df.copy()
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
    # loss == 0 with gain > 0 (pure uptrend) -> rs = inf -> RSI correctly resolves to 100.
    # loss == 0 and gain == 0 (no movement at all) -> rs = NaN -> RSI stays NaN, correctly "undefined".
    df["RSI"] = 100 - 100 / (1 + rs)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df
