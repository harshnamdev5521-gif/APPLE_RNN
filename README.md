# Apple Stock Price Prediction – Streamlit

This Streamlit application is based on the uploaded Apple Stock Price Prediction using RNN notebook.

## Features

- Introduction page
- AAPL daily dataset page
- Closing-price visualization
- Prediction date input
- Historical-date prediction using the previous 10 closing prices
- Future-date recursive forecasting
- RNN training controls
- Actual vs Predicted graph
- Training vs Validation Loss graph
- MAE, RMSE and R² metrics

## Run

1. Open PowerShell in this folder.
2. Install packages:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python -m streamlit run app.py
```

4. Enter your Alpha Vantage API key in the sidebar.

## Important

The notebook uses:
- AAPL
- Close price
- MinMaxScaler
- 10 time steps
- 80/20 train-test split
- SimpleRNN(50, activation='tanh')
- Dense(1)
- Adam optimizer
- Mean squared error loss
- 100 epochs
- batch size 8

The application keeps these core settings while adding an interactive Streamlit interface and prediction-date input.
