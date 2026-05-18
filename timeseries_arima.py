import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def save_plot(fig, filename):
    os.makedirs("Images", exist_ok=True)
    fig.savefig(os.path.join("Images", filename), bbox_inches="tight")
    plt.close(fig)


def plot_rolling_stats(ts, window=12, filename_prefix="rolling"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ts.plot(ax=ax, label='Original')
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    rolling_mean.plot(ax=ax, label=f'Rolling Mean ({window})')
    rolling_std.plot(ax=ax, label=f'Rolling Std ({window})')
    ax.legend()
    ax.set_title('Rolling Mean & Standard Deviation')
    save_plot(fig, f"{filename_prefix}_mean_std.png")


def adf_test(ts):
    result = adfuller(ts.dropna())
    adf_stat, pvalue = result[0], result[1]
    return adf_stat, pvalue


def select_pq(ts, nlags=24, thresh=0.2):
    acf_vals = acf(ts, nlags=nlags, fft=False)
    pacf_vals = pacf(ts, nlags=nlags)
    # Choose p as first significant lag in PACF, q as first significant lag in ACF
    p = 1
    q = 1
    for i, val in enumerate(pacf_vals[1:], start=1):
        if abs(val) > thresh:
            p = i
            break
    for i, val in enumerate(acf_vals[1:], start=1):
        if abs(val) > thresh:
            q = i
            break
    return int(p), int(q), acf_vals, pacf_vals


def run_arima_analysis(dataset_path=None):
    # Load cleaned dataset if available, otherwise original
    path_clean = dataset_path or "cleaned_walmart.csv"
    path_raw = "Walmart Dataset.csv"
    if os.path.exists(path_clean):
        df = pd.read_csv(path_clean, parse_dates=['Order Date'])
    elif os.path.exists(path_raw):
        df = pd.read_csv(path_raw, encoding='latin1', parse_dates=['Order Date'])
    else:
        print("No dataset found (cleaned_walmart.csv or Walmart Dataset.csv).")
        return None

    # Aggregate monthly sales
    df = df.dropna(subset=['Order Date', 'Sales'])
    df = df.sort_values('Order Date')
    df.set_index('Order Date', inplace=True)
    monthly_sales = df['Sales'].resample('ME').sum()

    # Plot rolling statistics
    plot_rolling_stats(monthly_sales, window=12, filename_prefix='monthly_rolling')
    print("-> Saved rolling mean/std plot to Images/monthly_rolling_mean_std.png")

    # ADF testing and differencing until stationary (max d=3)
    d = 0
    ts = monthly_sales.copy()
    adf_stat, pvalue = adf_test(ts)
    print(f"ADF test p-value (d={d}): {pvalue:.5f}")
    while pvalue > 0.05 and d < 3:
        d += 1
        ts = ts.diff().dropna()
        adf_stat, pvalue = adf_test(ts)
        print(f"ADF test p-value (d={d}): {pvalue:.5f}")

    print(f"Selected differencing order d = {d}")

    # Plot ACF and PACF for the stationary series
    ax_acf = plot_acf(ts, lags=24)
    save_plot(ax_acf.figure, 'acf_stationary.png')
    print("-> Saved ACF plot to Images/acf_stationary.png")

    ax_pacf = plot_pacf(ts, lags=24, method='ywm')
    save_plot(ax_pacf.figure, 'pacf_stationary.png')
    print("-> Saved PACF plot to Images/pacf_stationary.png")

    # Select p and q
    p, q, acf_vals, pacf_vals = select_pq(ts, nlags=24, thresh=0.2)
    print(f"Selected p={p}, q={q} (threshold=0.2 heuristic)")

    # Train/test split (last 12 periods as test if possible)
    if len(monthly_sales) < 24:
        test_periods = max(3, int(len(monthly_sales) * 0.2))
    else:
        test_periods = 12
    train = monthly_sales[:-test_periods]
    test = monthly_sales[-test_periods:]

    print(f"Train periods: {len(train)}, Test periods: {len(test)}")

    # Fit ARIMA on training data
    try:
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
    except Exception as e:
        print("ARIMA fit failed:", e)
        return None

    # Forecast
    forecast = model_fit.forecast(steps=len(test))

    # Evaluate
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"ARIMA({p},{d},{q}) RMSE on test: {rmse:.3f}")

    # Plot forecast vs actual
    fig, ax = plt.subplots(figsize=(10, 5))
    train.plot(ax=ax, label='Train')
    test.plot(ax=ax, label='Test')
    forecast.index = test.index
    forecast.plot(ax=ax, label='Forecast')
    ax.legend()
    ax.set_title(f'ARIMA({p},{d},{q}) Forecast vs Actual (RMSE={rmse:.2f})')
    save_plot(fig, 'arima_forecast.png')
    print("-> Saved forecast plot to Images/arima_forecast.png")

    return {
        "order": (p, d, q),
        "rmse": float(rmse),
        "train_periods": len(train),
        "test_periods": len(test),
    }


if __name__ == '__main__':
    run_arima_analysis()
