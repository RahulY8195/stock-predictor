import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    return df


@st.cache_data(show_spinner=False)
def fetch_info(ticker: str) -> dict:
    import base64
    import requests
    from urllib.parse import urlparse

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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    df = df.copy()
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def build_sequences(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def predict_future(model, scaler, last_window: np.ndarray, n_days: int) -> np.ndarray:
    scaled = scaler.transform(last_window.reshape(-1, 1)).flatten()
    current = list(scaled)
    w = len(last_window)
    preds = []
    for _ in range(n_days):
        x = np.array(current[-w:]).reshape(1, -1)
        p = model.predict(x)[0]
        preds.append(p)
        current.append(p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


@st.cache_resource(show_spinner=False)
def train_model(series_tuple: tuple, window: int, model_name: str):
    series = np.array(series_tuple)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    X, y = build_sequences(scaled, window)
    split = int(len(X) * 0.85)

    if model_name == "Gradient Boosting":
        m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
    else:
        m = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )

    m.fit(X[:split], y[:split])
    return m, scaler


with st.sidebar:
    st.title("📈 Stock Predictor")
    ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    period = st.selectbox("Historical data range", ["1y", "2y", "3y", "5y"], index=1)

    st.divider()
    model_name = st.radio(
        "Model",
        ["Gradient Boosting", "MLP Neural Network"],
        index=0,
        help=(
            "Gradient Boosting: fast tree-based ensemble. "
            "MLP: lightweight neural network. Both train in seconds."
        ),
    )

    st.divider()
    window = st.slider("Look-back window (days)", 20, 120, 60, step=10)
    forecast_days = st.slider("Forecast horizon (days)", 5, 60, 30, step=5)

    st.divider()
    run_btn = st.button("Train & Predict", type="primary", width="stretch")

info = fetch_info(ticker)
logo_b64 = info["logo_b64"]
company_name = info["name"]

logo_img = (
    f'<img src="data:image/png;base64,{logo_b64}" width="36" '
    f'style="border-radius:6px;object-fit:contain;vertical-align:middle;margin-right:8px"/>'
    if logo_b64 else ""
)
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
        <span style="font-size:2.2rem;font-weight:800;color:white">{ticker}</span>
        <span style="font-size:1.4rem;color:#aaaaaa;display:flex;align-items:center">
            {logo_img}{company_name}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not run_btn:
    st.info("Configure settings in the sidebar and click **Train & Predict** to start.")
    st.stop()

with st.spinner(f"Fetching {ticker} data…"):
    df = fetch_data(ticker, period)

if df.empty:
    st.error(f"No data found for **{ticker}**. Check the ticker symbol and try again.")
    st.stop()

df = add_indicators(df)
close_series = df["Close"].squeeze().values.astype(float)

col1, col2, col3, col4 = st.columns(4)
latest = close_series[-1]
prev = close_series[-2]
change = latest - prev
pct = change / prev * 100
col1.metric("Latest Close", f"${latest:.2f}", f"{change:+.2f} ({pct:+.2f}%)")
col2.metric("52-wk High", f"${close_series.max():.2f}")
col3.metric("52-wk Low", f"${close_series.min():.2f}")
col4.metric("Data points", f"{len(df):,}")

with st.spinner(f"Training {model_name}…"):
    model, scaler = train_model(tuple(close_series), window, model_name)

future_prices = predict_future(model, scaler, close_series[-window:], forecast_days)

scaled_full = scaler.transform(close_series.reshape(-1, 1)).flatten()
X_all, _ = build_sequences(scaled_full, window)
y_pred = scaler.inverse_transform(model.predict(X_all).reshape(-1, 1)).flatten()
y_true = close_series[window:]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

cm1, cm2, cm3 = st.columns(3)
cm1.metric("MAE", f"${mae:.2f}")
cm2.metric("RMSE", f"${rmse:.2f}")
cm3.metric("MAPE", f"{mape:.2f}%")

last_date = df.index[-1]
future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]

st.subheader("Price History & Forecast")
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
)

fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(), name="Actual",
    line=dict(color="#636EFA", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MA20"].squeeze(), name="MA 20",
    line=dict(color="#FFA15A", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MA50"].squeeze(), name="MA 50",
    line=dict(color="#00CC96", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index[window:], y=y_pred, name="Model fit",
    line=dict(color="#EF553B", width=1, dash="dash"), opacity=0.8), row=1, col=1)
fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name="Forecast",
    line=dict(color="#AB63FA", width=2), mode="lines+markers",
    marker=dict(size=4)), row=1, col=1)

upper = future_prices * 1.05
lower = future_prices * 0.95
fig.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(upper) + list(lower[::-1]),
    fill="toself", fillcolor="rgba(171,99,250,0.1)",
    line=dict(color="rgba(255,255,255,0)"), name="±5% band"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(), name="RSI",
    line=dict(color="#19D3F3", width=1)), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(), name="MACD",
    line=dict(color="#FF6692", width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Signal"].squeeze(), name="Signal",
    line=dict(color="#B6E880", width=1)), row=3, col=1)

fig.update_layout(
    height=700, template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="MACD", row=3, col=1)
st.plotly_chart(fig, width="stretch")

rsi_val = df["RSI"].squeeze().iloc[-1]
macd_val = df["MACD"].squeeze().iloc[-1]
signal_val = df["Signal"].squeeze().iloc[-1]
ma20_val = df["MA20"].squeeze().iloc[-1]
ma50_val = df["MA50"].squeeze().iloc[-1]
price_val = close_series[-1]

if rsi_val > 70:
    rsi_signal = "🔴 Overbought — price may pull back"
elif rsi_val < 30:
    rsi_signal = "🟢 Oversold — potential bounce"
else:
    rsi_signal = "🟡 Neutral zone"

if macd_val > signal_val:
    macd_signal = "🟢 Bullish crossover — upward momentum"
else:
    macd_signal = "🔴 Bearish crossover — downward momentum"

if price_val > ma20_val > ma50_val:
    ma_signal = "🟢 Bullish — price above both MAs"
elif price_val < ma20_val < ma50_val:
    ma_signal = "🔴 Bearish — price below both MAs"
else:
    ma_signal = "🟡 Mixed — watch for crossover"

st.subheader("Indicator Explanations")
c1, c2, c3, c4, c5 = st.columns(5)

c1.markdown("**MA 20 & MA 50**")
c1.caption(
    "Moving averages smooth out price noise. When MA 20 crosses above "
    "MA 50 it's a bullish 'golden cross'; below is a bearish 'death cross'."
)
c1.info(ma_signal)

c2.markdown("**RSI**")
c2.caption(
    "Relative Strength Index (0–100) measures momentum. Above 70 = "
    f"overbought, below 30 = oversold. Current: **{rsi_val:.1f}**"
)
c2.info(rsi_signal)

c3.markdown("**MACD**")
c3.caption(
    "Moving Average Convergence Divergence tracks trend direction. "
    "When the MACD line crosses above the Signal line, momentum is turning bullish."
)
c3.info(macd_signal)

c4.markdown("**Model Fit**")
c4.caption(
    "The dashed red line shows how well the model explains historical "
    "prices. Closer to actual = better trained model."
)

c5.markdown("**±5% Band**")
c5.caption(
    "Shaded forecast range showing ±5% uncertainty around predictions. "
    "Real price movement could exceed this — treat as a rough confidence zone, not a guarantee."
)

st.subheader("Forecast Table")
forecast_df = pd.DataFrame({
    "Date": future_dates.strftime("%Y-%m-%d"),
    "Predicted Close ($)": [f"{p:.2f}" for p in future_prices],
    "Change vs Today ($)": [f"{p - latest:+.2f}" for p in future_prices],
    "Change vs Today (%)": [f"{(p - latest)/latest*100:+.2f}%" for p in future_prices],
})
st.dataframe(forecast_df, width="stretch", hide_index=True)

with st.expander("Raw historical data"):
    st.dataframe(
        df[["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD"]].tail(100),
        width="stretch",
    )
