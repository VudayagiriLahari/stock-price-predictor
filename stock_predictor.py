
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import yfinance as yf
from datetime import datetime, timedelta
import os

#  DATA LOADING


def load_stock_data(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    """Download historical OHLCV data for the given ticker."""
    print(f"\n Downloading data for {ticker} ...")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    print(f"   {len(df)} trading days loaded  ({df.index[0].date()} → {df.index[-1].date()})")
    return df

# 2 FEATURE ENGINEERING


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as model features."""
    df = df.copy()

  
    df['MA_10']  = df['Close'].rolling(10).mean()
    df['MA_50']  = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()

   
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df['RSI'] = 100 - 100 / (1 + rs)

  
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

  
    rolling_std = df['Close'].rolling(20).std()
    df['BB_Width'] = (2 * rolling_std) / df['Close'].rolling(20).mean()

  
    for lag in [1, 3, 5, 10]:
        df[f'Close_Lag{lag}'] = df['Close'].shift(lag)

    df.dropna(inplace=True)
    return df



# 3 LINEAR REGRESSION MODEL


def linear_regression_model(df: pd.DataFrame):
    """Train & evaluate a Linear Regression model."""
    print("\n🔵 Training Linear Regression model ...")

    feature_cols = [
        'MA_10', 'MA_50', 'Volatility_20', 'RSI',
        'MACD', 'Signal', 'BB_Width',
        'Close_Lag1', 'Close_Lag3', 'Close_Lag5', 'Close_Lag10'
    ]
    X = df[feature_cols].values
    y = df['Close'].values

    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test       = df.index[split:]

    scaler   = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = _eval(y_test, y_pred, "Linear Regression")
    return dates_test, y_test, y_pred, metrics

# 4 LSTM MODEL


def lstm_model(df: pd.DataFrame, look_back: int = 60):
    """Train & evaluate an LSTM model on closing prices."""
    print("\n🟣 Training LSTM model ...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("  TensorFlow not available – skipping LSTM.")
        return None, None, None, None

    prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    split  = int(len(scaled) * 0.80)
    train  = scaled[:split]
    test   = scaled[split - look_back:]

    def make_sequences(data, lb):
        X, y = [], []
        for i in range(lb, len(data)):
            X.append(data[i - lb:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(train, look_back)
    X_test,  y_test  = make_sequences(test,  look_back)

    X_train = X_train.reshape(-1, look_back, 1)
    X_test  = X_test.reshape(-1, look_back, 1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=50, batch_size=32,
        validation_split=0.1,
        callbacks=[es], verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()


    dates_test = df.index[split + (len(scaled[split:]) - len(y_actual)):]

    metrics = _eval(y_actual, y_pred, "LSTM")
    return dates_test, y_actual, y_pred, metrics

# 5 METRICS


def _eval(actual, pred, name):
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2   = r2_score(actual, pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-9))) * 100

    print(f"   📊 {name} Results:")
    print(f"      MAE  = ${mae:.2f}")
    print(f"      RMSE = ${rmse:.2f}")
    print(f"      R²   = {r2:.4f}")
    print(f"      MAPE = {mape:.2f}%")
    return {"name": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# 6 VISUALISATIONS


def plot_results(df, ticker,
                 lr_dates, lr_actual, lr_pred, lr_metrics,
                 lstm_dates=None, lstm_actual=None, lstm_pred=None, lstm_metrics=None,
                 save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    has_lstm = lstm_dates is not None

    fig, axes = plt.subplots(3 if has_lstm else 2, 1,
                             figsize=(14, 18 if has_lstm else 12))
    fig.suptitle(f"{ticker} Stock Price Prediction", fontsize=18, fontweight='bold', y=0.98)

    # Full price history
    ax0 = axes[0]
    ax0.plot(df.index, df['Close'], color='steelblue', linewidth=1.2, label='Close Price')
    ax0.plot(df.index, df['MA_50'],  color='orange',    linewidth=1.0, linestyle='--', label='MA 50')
    ax0.plot(df.index, df['MA_200'], color='red',       linewidth=1.0, linestyle='--', label='MA 200')
    ax0.set_title('Historical Price with Moving Averages', fontsize=13)
    ax0.set_ylabel('Price (USD)')
    ax0.legend()
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Linear Regression predictions
    ax1 = axes[1]
    ax1.plot(lr_dates, lr_actual, color='steelblue', linewidth=1.5, label='Actual')
    ax1.plot(lr_dates, lr_pred,   color='tomato',    linewidth=1.5, linestyle='--', label='Predicted')
    ax1.fill_between(lr_dates, lr_actual, lr_pred, alpha=0.15, color='orange')
    ax1.set_title(f'Linear Regression  |  RMSE=${lr_metrics["RMSE"]:.2f}  R²={lr_metrics["R2"]:.3f}', fontsize=13)
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    # LSTM predictions
    if has_lstm:
        ax2 = axes[2]
        ax2.plot(lstm_dates, lstm_actual, color='steelblue', linewidth=1.5, label='Actual')
        ax2.plot(lstm_dates, lstm_pred,   color='mediumpurple', linewidth=1.5, linestyle='--', label='Predicted')
        ax2.fill_between(lstm_dates, lstm_actual, lstm_pred, alpha=0.15, color='mediumpurple')
        ax2.set_title(f'LSTM  |  RMSE=${lstm_metrics["RMSE"]:.2f}  R²={lstm_metrics["R2"]:.3f}', fontsize=13)
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(save_path, f"{ticker}_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n Plot saved → {path}")
    return path


def plot_residuals(lr_actual, lr_pred, ticker, save_path="results"):
    """Residual distribution for Linear Regression."""
    residuals = lr_actual - lr_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{ticker} – Linear Regression Residuals', fontsize=14)

    axes[0].scatter(range(len(residuals)), residuals, alpha=0.4, color='tomato', s=10)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_title('Residuals over time')
    axes[0].set_ylabel('Residual (USD)')

    axes[1].hist(residuals, bins=40, color='steelblue', edgecolor='white')
    axes[1].set_title('Residual Distribution')
    axes[1].set_xlabel('Residual (USD)')

    plt.tight_layout()
    path = os.path.join(save_path, f"{ticker}_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Residual plot saved → {path}")

# 7 NEXT-DAY FORECAST


def forecast_next_day(df: pd.DataFrame):
    """Simple next-day price forecast using lag features."""
    feature_cols = [
        'MA_10', 'MA_50', 'Volatility_20', 'RSI',
        'MACD', 'Signal', 'BB_Width',
        'Close_Lag1', 'Close_Lag3', 'Close_Lag5', 'Close_Lag10'
    ]
    X = df[feature_cols].values
    y = df['Close'].values

    scaler = MinMaxScaler()
    X_s    = scaler.fit_transform(X)

    model  = LinearRegression()
    model.fit(X_s, y)

    latest     = df[feature_cols].iloc[-1].values.reshape(1, -1)
    latest_s   = scaler.transform(latest)
    prediction = model.predict(latest_s)[0]

    print(f"\n🔮 Next-day price forecast: ${prediction:.2f}")
    print(f"   (Last closing price: ${df['Close'].iloc[-1]:.2f})")
    return prediction


# 8 MAIN


def main():
    TICKER = "AAPL"

    # Load 
    raw_df = load_stock_data(TICKER, period="5y")
    df     = add_features(raw_df)

    # Models
    lr_dates, lr_actual, lr_pred, lr_metrics = linear_regression_model(df)

    # LSTM 
    lstm_dates, lstm_actual, lstm_pred, lstm_metrics = lstm_model(df, look_back=60)

    # Plots
    plot_results(df, TICKER,
                 lr_dates, lr_actual, lr_pred, lr_metrics,
                 lstm_dates, lstm_actual, lstm_pred, lstm_metrics,
                 save_path="results")
    plot_residuals(lr_actual, lr_pred, TICKER, save_path="results")

    # Next-day prediction
    forecast_next_day(df)

    print("\n All done! Check the 'results/' folder for plots.")


if __name__ == "__main__":
    main()
