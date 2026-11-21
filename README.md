# ML-Trading

A learning-focused machine learning project for trading that predicts which barrier a stock will hit first—**stop loss**, **take profit (top profit)**, or **time limit**—given the three barrier parameters and the stock's past price/market evolution, using the **Alpaca Python** tool (`alpaca-py`).

## Overview & Goal

This project is built primarily as a hands-on learning project to get more comfortable working with machine learning models, data pipelines, and evaluation workflows in preparation for potential summer internships.

Instead of only predicting raw price direction or point returns, the model frames trade outcomes around three barriers:
- **Stop Loss:** Downside risk threshold to cap losses on a trade.
- **Take Profit (Top Profit):** Upside target threshold to lock in gains.
- **Time Limit:** Maximum holding window (in bars/time steps) before the trade expires.

Given a candidate set of these **3 barrier parameters** alongside **historical stock evolution** (past price action and derived features fetched via Alpaca), the model predicts which barrier will be hit first.

## How It Works

1. **Market Data via Alpaca Python:** Use the Alpaca Python SDK (`alpaca-py`) to pull historical stock market data and build lag, return, volatility, or technical features representing past stock evolution.
2. **Barrier Labeling:** Define stop-loss, take-profit, and time-limit barriers and label historical windows by which barrier is reached first.
3. **Model Training & Experimentation:** Train and compare machine learning models to predict barrier outcomes from historical stock evolution and the three barrier parameters, focusing on clean experimentation and evaluation.
4. **Barrier Evaluation:** Inspect predicted outcomes across barrier configurations to explore how barrier choices behave on historical data.

## Project Structure

```text
ML-Trading/
├── data/          # Raw and processed market data (keep large/sensitive data out of Git)
├── notebooks/     # Exploratory data analysis and ML experiments
├── src/           # Alpaca data fetching, feature engineering, labeling, and model code
├── README.md      # Project documentation
└── requirements.txt
