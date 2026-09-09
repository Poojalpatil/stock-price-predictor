"""
1. Downloads historical stock data
2. Creates features (technical indicators)
3. Creates the target (up/down tomorrow)
4. Trains a Random Forest model
5. Saves the trained model as model.pkl

"""

import yfinance as yf
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download stock data

TICKER = "RELIANCE.NS"   # change this to any stock, e.g. "TCS.NS", "AAPL", "INFY.NS"
print(f"Downloading data for {TICKER} ...")

df = yf.download(TICKER, period="2y")

if df.empty:
    raise ValueError("No data downloaded. Check the ticker symbol or your internet connection.")

# yfinance sometimes returns multi-level columns, flatten them
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"Downloaded {len(df)} rows of data.")


# Target = 1 if tomorrow's close is higher than today's close, else 0
df['Tomorrow'] = df['Close'].shift(-1)
df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)


# STEP 3: Create features

df['SMA_5'] = df['Close'].rolling(5).mean()
df['SMA_20'] = df['Close'].rolling(20).mean()
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Daily_Return'].rolling(5).std()


df = df.dropna()

FEATURES = ['SMA_5', 'SMA_20', 'Daily_Return', 'Volatility', 'Volume']
X = df[FEATURES]
y = df['Target']


# Train/test split (time-based, NOT random)

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")


# Train the model

model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split=10)
model.fit(X_train, y_train)


# Evaluate

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, predictions, target_names=["Down", "Up"]))


# Save the model

joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
print("You can now run the Streamlit app with: streamlit run app.py")