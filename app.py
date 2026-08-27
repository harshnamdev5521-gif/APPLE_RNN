
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Apple Stock Price Prediction",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0;
}
.subtitle {
    font-size: 18px;
    color: #666;
    margin-top: 0;
}
.card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    background: #fafafa;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Constants based on the uploaded notebook
# ---------------------------------------------------------
SYMBOL = "AAPL"
TIME_STEPS = 10
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 8


# ---------------------------------------------------------
# Data functions
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(api_key):
    """Load AAPL daily data using the same Alpha Vantage approach
    used in the uploaded notebook.
    """
    if not api_key:
        raise ValueError("Please enter an Alpha Vantage API key.")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": SYMBOL,
        "apikey": api_key,
        "outputsize": "compact",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "Time Series (Daily)" not in payload:
        message = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
        raise ValueError(message or "Alpha Vantage did not return daily stock data.")

    df = pd.DataFrame.from_dict(
        payload["Time Series (Daily)"],
        orient="index"
    )

    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df


def prepare_sequences(df):
    data = df[["Close"]].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(TIME_STEPS, len(scaled_data)):
        X.append(scaled_data[i - TIME_STEPS:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X) * 0.8)

    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    return scaler, X_train, X_test, y_train, y_test


def build_model():
    model = Sequential([
        SimpleRNN(
            50,
            activation="tanh",
            input_shape=(TIME_STEPS, 1)
        ),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )
    return model


@st.cache_resource(show_spinner=False)
def train_model(df, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
    scaler, X_train, X_test, y_train, y_test = prepare_sequences(df)

    model = build_model()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    return model, scaler, X_test, y_test, history


def predict_by_date(model, scaler, df, prediction_date):
    """For a historical/trading date, use the previous 10 available
    closing prices. For a future date, recursively predict one
    trading-day-at-a-time until the requested date.
    """
    target = pd.Timestamp(prediction_date).normalize()
    history_df = df[df.index.normalize() < target].copy()

    if len(history_df) < TIME_STEPS:
        return None, "Not enough historical data. At least 10 previous trading days are required."

    last_prices = history_df["Close"].values[-TIME_STEPS:].astype(float).tolist()

    # Historical date: one-step prediction, matching the notebook logic.
    if target <= df.index.max().normalize():
        scaled = scaler.transform(np.array(last_prices).reshape(-1, 1))
        X_input = scaled.reshape(1, TIME_STEPS, 1)
        prediction = model.predict(X_input, verbose=0)
        price = float(scaler.inverse_transform(prediction)[0][0])
        return price, "one-step prediction using the previous 10 available closing prices"

    # Future date: recursively predict business days.
    current_date = df.index.max().normalize()

    while current_date < target:
        next_day = current_date + pd.Timedelta(days=1)

        # Move to Monday when crossing a weekend.
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)

        scaled = scaler.transform(
            np.array(last_prices[-TIME_STEPS:]).reshape(-1, 1)
        )
        X_input = scaled.reshape(1, TIME_STEPS, 1)

        prediction = model.predict(X_input, verbose=0)
        predicted_price = float(
            scaler.inverse_transform(prediction)[0][0]
        )

        last_prices.append(predicted_price)
        current_date = next_day

    return last_prices[-1], "recursive forecast from the latest available market data"


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🍎 Apple Stock Predictor")
st.sidebar.caption("RNN / SimpleRNN based stock-price prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Introduction",
        "Dataset",
        "Visualization",
        "Prediction",
        "Model Performance",
    ],
)

api_key = st.sidebar.text_input(
    "Alpha Vantage API Key",
    value=os.getenv("ALPHAVANTAGE_API_KEY", ""),
    type="password",
    help="Enter your Alpha Vantage API key. It is used only to retrieve AAPL daily data."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Model architecture from the notebook: "
    "SimpleRNN(50, tanh) → Dense(1), with 10 time steps."
)


# ---------------------------------------------------------
# Introduction
# ---------------------------------------------------------
if page == "Introduction":
    st.markdown('<p class="main-title">🍎 Apple Stock Price Prediction Using RNN</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Interactive Streamlit web application based on the uploaded RNN notebook.</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📊 Input Data")
        st.write(
            "The application retrieves AAPL daily stock data and uses the "
            "Close price for the RNN prediction."
        )

    with c2:
        st.markdown("### 🧠 RNN Model")
        st.write(
            "The model uses a SimpleRNN layer with 50 units and tanh activation, "
            "followed by a single Dense output."
        )

    with c3:
        st.markdown("### 📅 Date Prediction")
        st.write(
            "Select a prediction date from the Prediction page. "
            "Historical dates use the previous 10 prices; future dates are forecast recursively."
        )

    st.markdown("## Project Workflow")
    st.write(
        "1. Fetch AAPL daily data → "
        "2. Select Close price → "
        "3. Min-Max scaling → "
        "4. Create 10-step time-series sequences → "
        "5. 80/20 train-test split → "
        "6. Train SimpleRNN → "
        "7. Predict and inverse-transform the price → "
        "8. Display the result."
    )

    st.warning(
        "Stock-price prediction is experimental and should not be treated as financial advice."
    )


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
elif page == "Dataset":
    st.title("📋 AAPL Dataset")

    if not api_key:
        st.info("Enter your Alpha Vantage API key in the sidebar to load the dataset.")
    else:
        try:
            df = load_stock_data(api_key)

            a, b, c, d = st.columns(4)
            a.metric("Symbol", SYMBOL)
            b.metric("Rows", f"{len(df):,}")
            c.metric("Start Date", df.index.min().strftime("%Y-%m-%d"))
            d.metric("Latest Date", df.index.max().strftime("%Y-%m-%d"))

            st.subheader("Latest 10 Records")
            st.dataframe(df.tail(10), use_container_width=True)

            st.subheader("Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)

        except Exception as e:
            st.error(f"Unable to load dataset: {e}")


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
elif page == "Visualization":
    st.title("📈 Apple Closing Price Visualization")

    if not api_key:
        st.info("Enter your Alpha Vantage API key in the sidebar.")
    else:
        try:
            df = load_stock_data(api_key)

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df.index, df["Close"])
            ax.set_title("Apple (AAPL) Closing Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Closing Price")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()

            st.pyplot(fig, use_container_width=True)

            st.subheader("Recent Closing Prices")
            st.line_chart(df["Close"].tail(120))

        except Exception as e:
            st.error(f"Unable to display visualization: {e}")


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
elif page == "Prediction":
    st.title("🔮 Predict Apple Stock Price")

    if not api_key:
        st.info("Enter your Alpha Vantage API key in the sidebar first.")
    else:
        try:
            df = load_stock_data(api_key)

            st.write(
                f"Latest available AAPL data in the application: "
                f"**{df.index.max().strftime('%Y-%m-%d')}**"
            )

            prediction_date = st.date_input(
                "📅 Select Prediction Date",
                value=(df.index.max() + pd.Timedelta(days=1)).date(),
                min_value=(df.index.min() + pd.Timedelta(days=TIME_STEPS)).date(),
                max_value=(df.index.max() + pd.Timedelta(days=365)).date(),
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                epochs = st.number_input(
                    "Training Epochs",
                    min_value=1,
                    max_value=300,
                    value=DEFAULT_EPOCHS,
                    step=10,
                )

            with col2:
                batch_size = st.selectbox(
                    "Batch Size",
                    [4, 8, 16, 32],
                    index=1,
                )

            if st.button("🚀 Predict Price", use_container_width=True):
                with st.spinner("Training the RNN and generating prediction..."):
                    model, scaler, X_test, y_test, history = train_model(
                        df,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                    )

                    price, explanation = predict_by_date(
                        model,
                        scaler,
                        df,
                        prediction_date,
                    )

                if price is None:
                    st.error(explanation)
                else:
                    st.success(f"Prediction generated using {explanation}.")
                    st.metric(
                        "Predicted AAPL Closing Price",
                        f"${price:,.2f}"
                    )

                    st.caption(
                        "This is a model forecast, not a guaranteed market price."
                    )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------------------------------------------------
# Model Performance
# ---------------------------------------------------------
elif page == "Model Performance":
    st.title("🧪 Model Performance")

    if not api_key:
        st.info("Enter your Alpha Vantage API key in the sidebar.")
    else:
        try:
            df = load_stock_data(api_key)

            with st.spinner("Training model for evaluation..."):
                model, scaler, X_test, y_test, history = train_model(df)

            y_pred = model.predict(X_test, verbose=0)

            y_pred_actual = scaler.inverse_transform(y_pred)
            y_test_actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            )

            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
            r2 = r2_score(y_test_actual, y_pred_actual)

            a, b, c = st.columns(3)
            a.metric("MAE", f"{mae:.4f}")
            b.metric("RMSE", f"{rmse:.4f}")
            c.metric("R²", f"{r2:.4f}")

            st.subheader("Actual vs Predicted")

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(y_test_actual, label="Actual Close Price")
            ax.plot(y_pred_actual, label="Predicted Close Price")
            ax.set_title("Apple Stock Price: Actual vs Predicted")
            ax.set_xlabel("Test Trading Days")
            ax.set_ylabel("Close Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)

            st.subheader("RNN Training vs Validation Loss")

            fig2, ax2 = plt.subplots(figsize=(14, 5))
            ax2.plot(history.history["loss"], label="Training Loss")
            ax2.plot(history.history["val_loss"], label="Validation Loss")
            ax2.set_title("RNN Training vs Validation Loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("MSE Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Could not calculate model performance: {e}")


st.markdown("---")
st.caption("Apple Stock Price Prediction • SimpleRNN • Streamlit")
