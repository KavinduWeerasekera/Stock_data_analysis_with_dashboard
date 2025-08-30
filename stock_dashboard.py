import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import time

st.set_page_config(layout="wide")
st.title("📈 Real-Time Stock Prediction Dashboard")

# Sidebar widgets
stock = st.sidebar.selectbox("Select a Stock", ["TSLA", "AAPL", "GOOG", "AMZN", "MSFT"])
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"])
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo"])

# --- Load stock data with caching (refresh every 5 min) ---
@st.cache_data(ttl=300)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.reset_index(inplace=True)
    datetime_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df[datetime_col] = pd.to_datetime(df[datetime_col]).dt.tz_localize(None)
    return df

df = load_data(stock, period, interval)

if df.empty:
    st.error("No data fetched for this combination. Please select different options.")
    st.stop()

st.subheader(f"📊 Live Price Data for {stock}")
st.dataframe(df.tail())

datetime_col = 'Datetime' if 'Datetime' in df.columns else 'Date'

# --- Candlestick chart ---
fig_candle = go.Figure(data=[go.Candlestick(
    x=df[datetime_col],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Candlestick'
)])
fig_candle.update_layout(title=f"🕯️ {stock} Stock Price", xaxis_rangeslider_visible=False)
st.plotly_chart(fig_candle, use_container_width=True)

# --- Prophet Model Training ---
df_prophet = df[[datetime_col, 'Close']].copy()
df_prophet.columns = ['ds', 'y']

model = Prophet()
model.fit(df_prophet)

# Forecast next hour and next day
future_hour = model.make_future_dataframe(periods=60, freq='min')
forecast_hour = model.predict(future_hour)

future_day = model.make_future_dataframe(periods=1440, freq='min')
forecast_day = model.predict(future_day)

pred_next_hour = forecast_hour.iloc[-1]['yhat']
pred_next_day = forecast_day.iloc[-1]['yhat']

# Stock direction
if len(df_prophet) >= 2:
    direction = "📈 UP" if df_prophet['y'].iloc[-1] > df_prophet['y'].iloc[-2] else "📉 DOWN"
else:
    direction = "N/A"

st.subheader("📊 Prediction Results")
col1, col2, col3 = st.columns(3)
col1.metric("📈 Direction", direction)
col2.metric("🕐 Next Hour Price", f"${pred_next_hour:.2f}")
col3.metric("📅 Next Day Price", f"${pred_next_day:.2f}")

# --- Forecast Plot ---
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=forecast_day['ds'], y=forecast_day['yhat'], name='Forecast'))
fig_forecast.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Actual'))
fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Price")
st.subheader("📉 Forecast Plot (Prophet)")
st.plotly_chart(fig_forecast, use_container_width=True)

# --- Accuracy on last 30 points ---
if len(df_prophet) >= 60:
    df_train = df_prophet[:-30]
    df_test = df_prophet[-30:]
    model_eval = Prophet()
    model_eval.fit(df_train)
    future_eval = model_eval.make_future_dataframe(periods=30)
    forecast_eval = model_eval.predict(future_eval)
    y_true = df_test['y'].values
    y_pred = forecast_eval.iloc[-30:]['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    col4, col5 = st.columns(2)
    col4.metric("📏 MAE", f"{mae:.2f}")
    col5.metric("📐 RMSE", f"{rmse:.2f}")
else:
    st.info("Not enough data to evaluate accuracy.")

st.info("⚠️ Dashboard retrains every 5 minutes automatically.")

# --- Auto-retrain logic ---
st_autorefresh_interval = 300  # 5 minutes
st.session_state.setdefault("last_rerun", time.time())

if time.time() - st.session_state["last_rerun"] > st_autorefresh_interval:
    st.session_state["last_rerun"] = time.time()
    st.experimental_rerun()
