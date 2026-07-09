import base64
from datetime import date, timedelta
from urllib.parse import urlparse

import pandas as pd
import requests
import yfinance as yf


def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def fetch_info(ticker: str) -> dict:
    result = {"logo_b64": "", "name": ticker}
    try:
        info = yf.Ticker(ticker).info
        result["name"] = info.get("shortName", ticker)
        website = info.get("website", "")
        if not website:
            return result
        domain = urlparse(website).netloc.replace("www.", "")
        logo_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
        resp = requests.get(logo_url, timeout=5)
        if resp.status_code == 200 and resp.content:
            result["logo_b64"] = base64.b64encode(resp.content).decode()
    except Exception:
        pass
    return result


def fetch_close_on_or_after(ticker: str, on_date: date) -> float | None:
    df = yf.download(
        ticker,
        start=on_date,
        end=on_date + timedelta(days=7),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None
    return float(df["Close"].iloc[0])
