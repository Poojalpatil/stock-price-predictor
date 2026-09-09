import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Stock Movement Predictor", page_icon="📈", layout="centered")

st.title("📈 Stock Price Movement Predictor")
st.write(
    "This app predicts whether a stock's closing price will go **UP or DOWN** "
    "the next trading day, using a Random Forest model trained on historical "
    "price data and technical indicators."
)

FEATURES = ['SMA_5', 'SMA_20', 'Daily_Return', 'Volatility', 'Volume']

# Company list (name shown to user -> ticker used for lookup)
# Add or remove companies here as you like

COMPANIES = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Wipro": "WIPRO.NS",
    "ITC": "ITC.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google (Alphabet)": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
}


# Load the trained model

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "model.pkl not found. Please run `python train.py` first to train and save the model."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


st.subheader("Choose a company")

selection_mode = st.radio(
    "How do you want to pick a stock?",
    ["Select from list", "Type my own ticker"],
    horizontal=True
)

if selection_mode == "Select from list":
    company_name = st.selectbox(
        "Company",
        list(COMPANIES.keys()),
        index=None,
        placeholder="Choose a company"
    )
    if company_name is None:
        st.info("Please select a company above to continue.")
        st.stop()
    ticker = COMPANIES[company_name]
    st.caption(f"Ticker symbol: `{ticker}`")
else:
    ticker = st.text_input(
        "Enter a stock ticker (NSE stocks need .NS, e.g. RELIANCE.NS, TCS.NS; US stocks e.g. AAPL, MSFT)",
        ""
    )
    if not ticker:
        st.info("Please enter a ticker above to continue.")
        st.stop()

st.subheader("Choose time range")

PERIOD_OPTIONS = {
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
    "Max (all available data)": "max",
}

period_label = st.selectbox(
    "How many years of data do you want to see?",
    list(PERIOD_OPTIONS.keys()),
    index=None,
    placeholder="Choose a time range"
)

if period_label is None:
    st.info("Please select a time range above to continue.")
    st.stop()

period = PERIOD_OPTIONS[period_label]

if st.button("Predict"):
    with st.spinner(f"Fetching {period_label.lower()} of data for {ticker} ..."):
        data = yf.download(ticker, period=period)

    if data.empty:
        st.error("No data found for this ticker. Please check the symbol and try again.")
        st.stop()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Build the same features used in training
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(5).std()
    data = data.dropna()

    if data.empty:
        st.error("Not enough historical data to compute indicators for this ticker.")
        st.stop()

    latest_row = data[FEATURES].iloc[-1:]

    prediction = model.predict(latest_row)[0]
    probability = model.predict_proba(latest_row)[0]

    st.subheader("Prediction for the next trading day:")
    if prediction == 1:
        st.success(f"📈 UP  (confidence: {probability[1]:.1%})")
    else:
        st.error(f"📉 DOWN  (confidence: {probability[0]:.1%})")

    st.subheader(f"Price history ({period_label})")
    st.line_chart(data['Close'])

    st.subheader("Recent data used for prediction")
    st.dataframe(data[['Close'] + FEATURES].tail(10))

st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This project is for educational purposes only. "
    "Predictions are not financial advice and should not be used for real trading decisions."
)