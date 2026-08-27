#  Stock Price Predictor

A machine-learning project that predicts stock prices using historical market data, technical indicators, **Linear Regression**, and an **LSTM neural network**.

---

## Features

| Feature | Detail |
|---------|--------|
| Data Source | Yahoo Finance via `yfinance` |
| Models | Linear Regression + LSTM |
| Indicators | MA-10/50/200, RSI, MACD, Bollinger Bands, Volatility |
| Metrics | MAE, RMSE, R², MAPE |
| Output | Prediction plots + residual analysis + next-day forecast |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/stock-price-predictor.git
cd stock-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run
python stock_predictor.py
```

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
yfinance
tensorflow
```

---

## How It Works

### 1. Data Collection
Downloads 5 years of daily OHLCV data from Yahoo Finance for the chosen ticker (default: `AAPL`).

### 2. Feature Engineering
Computes 11 technical indicators:
- **Moving Averages** (10, 50, 200 day)
- **RSI** (Relative Strength Index)
- **MACD** + Signal line
- **Bollinger Band width**
- **Daily return volatility**
- **Lag features** (1, 3, 5, 10 days)

### 3. Models

#### Linear Regression (Baseline)
- 80/20 train-test split
- Features scaled with MinMaxScaler
- Fast and interpretable

#### LSTM (Advanced)
- 60-day look-back window
- 2-layer LSTM with Dropout
- EarlyStopping to prevent overfitting

### 4. Evaluation Metrics
- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coefficient of Determination
- **MAPE** – Mean Absolute Percentage Error

---

##  Output

```
results/
├── AAPL_predictions.png   # Actual vs predicted for both models
└── AAPL_residuals.png     # Residual analysis
```

---

##  Configuration

Change the ticker or time period inside `main()`:

```python
TICKER = "TSLA"   # Any Yahoo Finance ticker
raw_df = load_stock_data(TICKER, period="3y")
```

---

##  Sample Output

The script generates:
1. Historical price chart with MA overlays
2. Linear Regression predictions vs actual
3. LSTM predictions vs actual
4. Residual scatter + distribution plot
5. Next-day price forecast printed to console

---

##  Disclaimer

This project is for **educational purposes only**. Stock price predictions should **not** be used for real financial decisions.

---

##  Tech Stack

`Python` · `scikit-learn` · `TensorFlow/Keras` · `yfinance` · `Pandas` · `Matplotlib`
