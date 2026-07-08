import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API_BASE_URL = os.environ.get("STOCK_PREDICTOR_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")


def api_get(path: str, **params):
    resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json: dict | None = None):
    resp = requests.post(f"{API_BASE_URL}{path}", json=json, timeout=120)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False)
def fetch_info(ticker: str) -> dict:
    return api_get(f"/info/{ticker}")


@st.cache_data(show_spinner=False)
def fetch_stock(ticker: str, period: str) -> dict:
    return api_get(f"/stock/{ticker}", period=period)


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

try:
    info = fetch_info(ticker)
except requests.exceptions.RequestException:
    st.error(
        f"Can't reach the API at `{API_BASE_URL}`. Start it with "
        "`uvicorn app.main:app --reload` and refresh."
    )
    st.stop()

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
    try:
        stock = fetch_stock(ticker, period)
    except requests.exceptions.HTTPError:
        st.error(f"No data found for **{ticker}**. Check the ticker symbol and try again.")
        st.stop()

history = pd.DataFrame(stock["history"])
history["Date"] = pd.to_datetime(history["Date"])
history = history.set_index("Date")
close_series = history["Close"].values.astype(float)

col1, col2, col3, col4 = st.columns(4)
latest = close_series[-1]
prev = close_series[-2]
change = latest - prev
pct = change / prev * 100
col1.metric("Latest Close", f"${latest:.2f}", f"{change:+.2f} ({pct:+.2f}%)")
col2.metric("52-wk High", f"${close_series.max():.2f}")
col3.metric("52-wk Low", f"${close_series.min():.2f}")
col4.metric("Data points", f"{len(history):,}")

with st.spinner(f"Training {model_name}…"):
    prediction = api_post(
        "/predict",
        json={
            "ticker": ticker,
            "period": period,
            "model": model_name,
            "window": window,
            "forecast_days": forecast_days,
        },
    )

forecast_df_raw = pd.DataFrame(prediction["forecast"])
future_dates = pd.to_datetime(forecast_df_raw["date"])
future_prices = forecast_df_raw["predicted_price"].values

cm1, cm2, cm3 = st.columns(3)
cm1.metric("Historical fit MAE", f"${prediction['fit_mae']:.2f}")
cm2.metric("Historical fit RMSE", f"${prediction['fit_rmse']:.2f}")
cm3.metric("Historical fit MAPE", f"{prediction['fit_mape']:.2f}%")
st.caption(
    "These are **in-sample** metrics — how well the model explains the price history "
    "it was trained on. See **Tracked Accuracy** below for real out-of-sample performance."
)

st.subheader("Price History & Forecast")
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
)

fig.add_trace(go.Scatter(x=history.index, y=history["Close"], name="Actual",
    line=dict(color="#636EFA", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=history.index, y=history["MA20"], name="MA 20",
    line=dict(color="#FFA15A", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=history.index, y=history["MA50"], name="MA 50",
    line=dict(color="#00CC96", width=1, dash="dot")), row=1, col=1)
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

fig.add_trace(go.Scatter(x=history.index, y=history["RSI"], name="RSI",
    line=dict(color="#19D3F3", width=1)), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

fig.add_trace(go.Scatter(x=history.index, y=history["MACD"], name="MACD",
    line=dict(color="#FF6692", width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=history.index, y=history["Signal"], name="Signal",
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

rsi_val = history["RSI"].iloc[-1]
macd_val = history["MACD"].iloc[-1]
signal_val = history["Signal"].iloc[-1]
ma20_val = history["MA20"].iloc[-1]
ma50_val = history["MA50"].iloc[-1]
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
    "Historical fit MAE/RMSE/MAPE above show how well the model explains "
    "prices it was trained on — a starting sanity check, not proof it forecasts well."
)

c5.markdown("**±5% Band**")
c5.caption(
    "Shaded forecast range showing ±5% uncertainty around predictions. "
    "Real price movement could exceed this — treat as a rough confidence zone, not a guarantee."
)

st.subheader("Forecast Table")
forecast_table = pd.DataFrame({
    "Date": future_dates.dt.strftime("%Y-%m-%d"),
    "Predicted Close ($)": [f"{p:.2f}" for p in future_prices],
    "Change vs Today ($)": [f"{p - latest:+.2f}" for p in future_prices],
    "Change vs Today (%)": [f"{(p - latest)/latest*100:+.2f}%" for p in future_prices],
})
st.dataframe(forecast_table, width="stretch", hide_index=True)

st.subheader("Tracked Accuracy (Out-of-Sample)")
st.caption(
    "Every prediction made above is logged with its target date. Once that date "
    "passes, the app checks the real closing price and compares it to what was "
    "predicted — this is the only trustworthy accuracy number on this page."
)
acc_col, btn_col = st.columns([4, 1])
with btn_col:
    if st.button("Check for resolved predictions"):
        with st.spinner("Reconciling logged predictions with actual prices…"):
            api_post("/reconcile")

try:
    accuracy = api_get(f"/accuracy/{ticker}")
except requests.exceptions.RequestException:
    accuracy = None

if accuracy and accuracy["resolved_predictions"] > 0:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Resolved predictions", accuracy["resolved_predictions"])
    a2.metric("Out-of-sample MAE", f"${accuracy['mae']:.2f}")
    a3.metric("Out-of-sample RMSE", f"${accuracy['rmse']:.2f}")
    a4.metric("Out-of-sample MAPE", f"{accuracy['mape']:.2f}%")
else:
    st.info(
        "No resolved predictions yet for this ticker. Predictions logged today "
        "resolve once their target date arrives — check back after a trading day passes, "
        "or click **Check for resolved predictions** above."
    )

with st.expander("Prediction log"):
    try:
        log_rows = api_get(f"/predictions/{ticker}")
    except requests.exceptions.RequestException:
        log_rows = []
    if log_rows:
        log_df = pd.DataFrame(log_rows)[
            ["made_on", "target_date", "model_name", "window", "predicted_price", "actual_price"]
        ]
        st.dataframe(log_df, width="stretch", hide_index=True)
    else:
        st.caption("No predictions logged yet.")

with st.expander("Raw historical data"):
    st.dataframe(
        history[["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD"]].tail(100),
        width="stretch",
    )
