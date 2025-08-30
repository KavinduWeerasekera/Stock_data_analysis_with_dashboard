#If packages are not installed install using following line
#pip install streamlit yfinance pandas prophet plotly scikit-learn
#to run the file use  "streamlit run 16076_stock_dashboard.py"

import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression

import datetime
import time


# Auto-refresh every 5 minutes (300 seconds) using session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 300:  # 300 seconds = 5 minutes
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()


# obtaining stock data for tesla through yahoofinanace
ticker = 'TSLA'

# Downloading data 
data = yf.download(tickers=ticker, period='1d', interval='1m')  
print(data.tail())

def get_stock_direction(data):
    close_prices = data['Close']['TSLA']
    if close_prices.iloc[-1] > close_prices.iloc[-2]:
        return 'UP'
    else:
        return 'DOWN'

direction = get_stock_direction(data)
print("Current Stock Direction:", direction)

# Prepare data for Prophet
df = data.reset_index()[['Datetime', 'Close']]
df.columns = ['ds', 'y']

# Removing timezone information from 'ds' column
df['ds'] = df['ds'].dt.tz_localize(None)

# Initialize and fit Prophet model
model = Prophet()
model.fit(df)

# Forecast for next hour and next day
future = model.make_future_dataframe(periods=60, freq='min')  # Next 60 minutes
forecast_hour = model.predict(future)

future_day = model.make_future_dataframe(periods=1440, freq='min')  # Next 24 hours
forecast_day = model.predict(future_day)

# Get next hour prediction (last row of forecast_hour)
predicted_next_hour = forecast_hour.iloc[-1]['yhat']
# Get next day prediction (last row of forecast_day)
predicted_next_day = forecast_day.iloc[-1]['yhat']

print(f"Predicted Price in Next Hour: ${predicted_next_hour:.2f}")
print(f"Predicted Price in Next Day: ${predicted_next_day:.2f}")


import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(layout="wide")
import time



st.title("📈 Real-Time Stock Prediction Dashboard")

# Sidebar widgets
stock = st.sidebar.selectbox("Select a Stock", ["TSLA", "AAPL", "GOOG", "AMZN", "MSFT"])
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"])
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo"])

@st.cache_data(ttl=60)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    # Fix multi-index columns (drop second level if exists)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.reset_index(inplace=True)
    # Remove timezone info from datetime column
    datetime_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df[datetime_col] = pd.to_datetime(df[datetime_col]).dt.tz_localize(None)
    return df

df = load_data(stock, period, interval)

if df.empty:
    st.error("No data fetched for this combination of period and interval. Please select different options.")
    st.stop()

st.subheader(f"Live Price Data for {stock}")
st.dataframe(df.tail())

# Candlestick chart
datetime_col = 'Datetime' if 'Datetime' in df.columns else 'Date'

fig = go.Figure(data=[go.Candlestick(
    x=df[datetime_col],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Candlestick'
)])
fig.update_layout(title=f"{stock} Stock Price", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Prepare data for Prophet model
df_prophet = df[[datetime_col, 'Close']].copy()
df_prophet.columns = ['ds', 'y']
df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)

# Fit Prophet model
model = Prophet()
model.fit(df_prophet)

# Forecast next hour (60 mins) and next day (1440 mins)
future_hour = model.make_future_dataframe(periods=60, freq='min')
forecast_hour = model.predict(future_hour)

future_day = model.make_future_dataframe(periods=1440, freq='min')
forecast_day = model.predict(future_day)

# Predictions
predicted_next_hour = forecast_hour.iloc[-1]['yhat']
predicted_next_day = forecast_day.iloc[-1]['yhat']

# Determine stock direction (up/down) based on last two closes
if len(df_prophet) >= 2:
    direction = "UP" if df_prophet['y'].iloc[-1] > df_prophet['y'].iloc[-2] else "DOWN"
else:
    direction = "N/A"

st.subheader("📊 Prediction Results")
col1, col2, col3 = st.columns(3)
col1.metric("📈 Direction", direction)
col2.metric("🕐 Next Hour Price", f"${predicted_next_hour:.2f}")
col3.metric("📅 Next Day Price", f"${predicted_next_day:.2f}")

# Forecast plot with actual data
st.subheader("📉 Forecast Plot (Prophet)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast_day['ds'], y=forecast_day['yhat'], name='Forecast'))
fig2.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Actual'))
fig2.update_layout(xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig2, use_container_width=True)

# Accuracy evaluation on historical data (last 30 points)
def evaluate_accuracy(df):
    if len(df) < 60:
        return None, None  # Not enough data for split
    model = Prophet()
    df_train = df[:-30]
    df_test = df[-30:]
    model.fit(df_train)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    y_true = df_test['y'].values
    y_pred = forecast.iloc[-30:]['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

mae, rmse = evaluate_accuracy(df_prophet)

st.subheader("📐 Forecast Accuracy")
if mae is not None and rmse is not None:
    col4, col5 = st.columns(2)
    col4.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    col5.metric("RMSE (Root Mean Square Error)", f"{rmse:.2f}")
else:
    st.info("Not enough data to evaluate accuracy (need at least 60 data points).")

st.info("Note: Accuracy calculated from last 30 data points.")
