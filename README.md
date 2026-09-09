# 📈 Stock Price Movement Predictor

A machine learning project that predicts whether a stock's closing price will go
up or down the next trading day, using historical price data and technical
indicators. Built as an ML internship project.

## Features
- Downloads live/historical stock data using `yfinance`
- Engineers technical indicators: SMA-5, SMA-20, daily returns, volatility
- Trains a Random Forest Classifier to predict next-day price direction
- Interactive Streamlit web app for live predictions on any stock ticker

## Tech Stack
Python, scikit-learn, pandas, yfinance, Streamlit


## Model Details
Algorithm: Random Forest Classifier
Training data: 2 years of historical daily price data
Features: SMA-5, SMA-20, daily return, 5-day volatility, volume
Target: Binary classification — will tomorrow's close be higher than today's close?
Evaluation: Accuracy, precision, recall (time-based train/test split, no shuffling, to avoid data leakage)

## Disclaimer
This project is for educational purposes only and should not be used for real trading decisions.