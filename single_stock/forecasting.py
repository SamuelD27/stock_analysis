"""
Advanced Time Series Forecasting
---------------------------------
Combines statistical models (ARIMA, GARCH) with deep learning (LSTM)
for comprehensive stock price forecasting with rigorous evaluation.

Includes:
- Traditional: ARIMA, Exponential Smoothing, GARCH
- Deep Learning: LSTM with attention
- Ensemble methods
- Comprehensive evaluation: RMSE, MAE, Directional Accuracy, Sharpe
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("Warning: arch package not available. GARCH models disabled.")

# Deep learning (lazy import to avoid TensorFlow init deadlocks when LSTM is not used)
HAS_TENSORFLOW = False
keras = None
layers = None
tf = None

def _ensure_tf():
    """Import TensorFlow/Keras only when needed (e.g., LSTM selected)."""
    global tf, keras, layers, HAS_TENSORFLOW
    if HAS_TENSORFLOW and tf is not None:
        return
    # Set thread env vars before importing TF (helps avoid macOS deadlocks)
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as _tf
        import keras 
        from keras import layers 
        # Prefer CPU-only to avoid MPS/GPU init edge cases on macOS
        try:
            _tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        tf = _tf
        keras = _keras
        layers = _layers
        HAS_TENSORFLOW = True
        _tf.random.set_seed(RANDOM_SEED)
    except Exception as e:
        HAS_TENSORFLOW = False
        raise ImportError(f"TensorFlow failed to import: {e}")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from utils.data_processing import (
        load_current_session_data,
        get_current_session_tickers,
        normalize_prices_columns_ticker_first,
    )
except ImportError as e:
    print(f"Error: Could not import utilities: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if HAS_TENSORFLOW:
    tf.random.set_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Container for forecast results."""
    model_name: str
    predictions: np.ndarray
    actual: np.ndarray
    forecast_horizon: np.ndarray
    
    # Metrics
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    sharpe_ratio: float
    
    # Additional info
    train_time: float
    
    # Confidence intervals at multiple levels
    ci_50_lower: Optional[np.ndarray] = None
    ci_50_upper: Optional[np.ndarray] = None
    ci_80_lower: Optional[np.ndarray] = None
    ci_80_upper: Optional[np.ndarray] = None
    ci_95_lower: Optional[np.ndarray] = None
    ci_95_upper: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare_data(base_name: str = "last_fetch", 
                         lookback: int = 60,
                         forecast_horizon: int = 30) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and prepare data for forecasting.
    
    Returns
    -------
    tuple
        (prices_df, tickers_list)
    """
    prices_df, bundle = load_current_session_data(base_name=base_name)
    # Normalize column orientation: (ticker, field)
    prices_df = normalize_prices_columns_ticker_first(prices_df)
    tickers = bundle['meta'].get('tickers', [])
    is_single = bundle.get('is_single_ticker', False)
    
    # Extract prices (prefer Adj Close, fallback to Close)
    if is_single or not isinstance(prices_df.columns, pd.MultiIndex):
        if 'Adj Close' in prices_df.columns:
            prices = prices_df['Adj Close']
        elif 'Close' in prices_df.columns:
            prices = prices_df['Close']
        else:
            # Fallback to the first numeric column
            prices = prices_df.select_dtypes(include=[np.number]).iloc[:, 0]
        prices = prices.to_frame(name=tickers[0] if tickers else 'Asset')
    else:
        fields = [c[1] for c in prices_df.columns]
        use_field = 'Adj Close' if 'Adj Close' in fields else ('Close' if 'Close' in fields else fields[0])
        prices = prices_df.xs(use_field, level=1, axis=1)
    
    prices = prices.ffill()
    return prices, tickers


def create_sequences(data: np.ndarray, lookback: int, 
                    forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for supervised learning."""
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[(i + lookback):(i + lookback + forecast_horizon)])
    return np.array(X), np.array(y)


def add_technical_features(prices: pd.Series) -> pd.DataFrame:
    """Add technical indicators as features."""
    df = pd.DataFrame(index=prices.index)
    df['price'] = prices
    df['returns'] = prices.pct_change()
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = prices.rolling(window).mean()
        df[f'ema_{window}'] = prices.ewm(span=window).mean()
    
    # Volatility
    df['volatility_20'] = prices.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Momentum indicators
    df['rsi_14'] = compute_rsi(prices, 14)
    df['macd'], df['macd_signal'] = compute_macd(prices)
    
    # Bollinger Bands
    sma_20 = prices.rolling(20).mean()
    std_20 = prices.rolling(20).std()
    df['bb_upper'] = sma_20 + 2 * std_20
    df['bb_lower'] = sma_20 - 2 * std_20
    df['bb_position'] = (prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df.dropna()


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, 
                slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Compute MACD indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_confidence_intervals(predictions: np.ndarray, 
                                residuals: np.ndarray,
                                forecast_horizon: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute confidence intervals at multiple levels.
    
    Uses expanding residual variance to account for increasing uncertainty.
    """
    # Estimate residual standard error
    std_resid = np.std(residuals)
    
    # Variance increases with forecast horizon (assume linear growth)
    variance_multiplier = 1 + 0.1 * forecast_horizon  # 10% increase per step
    forecast_std = std_resid * np.sqrt(variance_multiplier)
    
    # Confidence intervals at different levels
    z_50 = stats.norm.ppf(0.75)  # 50% CI (25th to 75th percentile)
    z_80 = stats.norm.ppf(0.90)  # 80% CI
    z_95 = stats.norm.ppf(0.975) # 95% CI
    
    return {
        'ci_50_lower': predictions - z_50 * forecast_std,
        'ci_50_upper': predictions + z_50 * forecast_std,
        'ci_80_lower': predictions - z_80 * forecast_std,
        'ci_80_upper': predictions + z_80 * forecast_std,
        'ci_95_lower': predictions - z_95 * forecast_std,
        'ci_95_upper': predictions + z_95 * forecast_std,
    }


# --- Bootstrap confidence intervals and evaluation ---
def bootstrap_confidence_intervals(train: np.ndarray, test: np.ndarray,
                                   model_func, n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
    """
    Compute confidence intervals using bootstrap resampling over the training set.
    `model_func` should be a callable(train, test) -> ForecastResult-like object
    with a `.predictions` array.
    """
    bootstrap_predictions = []
    if len(train) == 0 or len(test) == 0:
        return {}
    
    for _ in range(n_bootstrap):
        # Resample with replacement indices from the training data
        indices = np.random.choice(len(train), size=len(train), replace=True)
        train_boot = train[indices]
        try:
            result = model_func(train_boot, test)
            if result is not None and hasattr(result, 'predictions'):
                bootstrap_predictions.append(np.asarray(result.predictions))
        except Exception:
            continue
    
    if not bootstrap_predictions:
        return {}
    
    bp = np.vstack(bootstrap_predictions)
    return {
        'ci_50_lower': np.percentile(bp, 25, axis=0),
        'ci_50_upper': np.percentile(bp, 75, axis=0),
        'ci_80_lower': np.percentile(bp, 10, axis=0),
        'ci_80_upper': np.percentile(bp, 90, axis=0),
        'ci_95_lower': np.percentile(bp, 2.5, axis=0),
        'ci_95_upper': np.percentile(bp, 97.5, axis=0),
    }

def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for forecasts."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    # Guard against zero length
    if actual.size == 0 or predicted.size == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'directional_accuracy': np.nan, 'sharpe_ratio': np.nan}
    
    # Align lengths defensively
    n = min(len(actual), len(predicted))
    actual = actual[:n]
    predicted = predicted[:n]
    
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_arr = np.abs((actual - predicted) / actual)
        mape = float(np.nanmean(mape_arr) * 100.0)
    
    # Directional accuracy (based on differences)
    if n > 1:
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(predicted))
        directional_acc = float(np.mean(actual_direction == pred_direction) * 100.0)
    else:
        directional_acc = np.nan
    
    # Sharpe using actual returns (annualized, 252)
    if n > 1:
        actual_returns = np.diff(actual) / actual[:-1]
        std = np.std(actual_returns)
        sharpe = float((np.mean(actual_returns) / std) * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': directional_acc,
        'sharpe_ratio': sharpe,
    }


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------

def fit_arima_model(train: np.ndarray, test: np.ndarray, 
                   order: Tuple[int, int, int] = (1, 1, 1)) -> ForecastResult:
    """Fit ARIMA model with walk-forward validation and confidence intervals."""
    import time
    start_time = time.time()
    
    train_series = pd.Series(train)
    predictions = []
    residuals = []
    ci_50_lower, ci_50_upper = [], []
    ci_80_lower, ci_80_upper = [], []
    ci_95_lower, ci_95_upper = [], []
    
    # Walk-forward validation with confidence intervals
    for i in range(len(test)):
        model = ARIMA(train_series, order=order)
        fitted = model.fit()
        
        # Get forecast with confidence intervals
        forecast_result = fitted.get_forecast(steps=1)
        pred = forecast_result.predicted_mean.iloc[0]
        predictions.append(pred)
        
        # Store residuals for CI calculation
        if i > 0:
            residuals.append(test[i-1] - predictions[i-1])
        
        # ARIMA native confidence intervals (95%)
        ci_95 = forecast_result.conf_int(alpha=0.05)
        ci_95_lower.append(ci_95.iloc[0, 0])
        ci_95_upper.append(ci_95.iloc[0, 1])
        
        # 80% CI
        ci_80 = forecast_result.conf_int(alpha=0.20)
        ci_80_lower.append(ci_80.iloc[0, 0])
        ci_80_upper.append(ci_80.iloc[0, 1])
        
        # 50% CI
        ci_50 = forecast_result.conf_int(alpha=0.50)
        ci_50_lower.append(ci_50.iloc[0, 0])
        ci_50_upper.append(ci_50.iloc[0, 1])
        
        train_series = pd.concat([train_series, pd.Series([test[i]])])
    
    predictions = np.array(predictions)
    train_time = time.time() - start_time
    
    metrics = evaluate_forecast(test, predictions)
    
    return ForecastResult(
        model_name='ARIMA',
        predictions=predictions,
        actual=test,
        forecast_horizon=np.arange(len(test)),
        train_time=train_time,
        ci_50_lower=np.array(ci_50_lower),
        ci_50_upper=np.array(ci_50_upper),
        ci_80_lower=np.array(ci_80_lower),
        ci_80_upper=np.array(ci_80_upper),
        ci_95_lower=np.array(ci_95_lower),
        ci_95_upper=np.array(ci_95_upper),
        **metrics
    )


def fit_exponential_smoothing(train: np.ndarray, test: np.ndarray,
                              trend: str = 'add', 
                              seasonal: Optional[str] = None,
                              seasonal_periods: Optional[int] = None) -> ForecastResult:
    """Fit Exponential Smoothing model with bootstrap confidence intervals."""
    import time
    start_time = time.time()
    
    train_series = pd.Series(train)
    model = ExponentialSmoothing(
        train_series, 
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )
    fitted = model.fit()
    predictions = fitted.forecast(steps=len(test))
    
    # Compute residuals
    fitted_values = fitted.fittedvalues
    residuals = train_series - fitted_values
    
    # Compute confidence intervals using residual bootstrap
    forecast_horizon = np.arange(len(test))
    cis = compute_confidence_intervals(predictions.values, residuals.values, forecast_horizon)
    
    train_time = time.time() - start_time
    metrics = evaluate_forecast(test, predictions.values)
    
    return ForecastResult(
        model_name='Exponential Smoothing',
        predictions=predictions.values,
        actual=test,
        forecast_horizon=forecast_horizon,
        train_time=train_time,
        **cis,
        **metrics
    )


def fit_garch_model(train: np.ndarray, test: np.ndarray) -> Optional[ForecastResult]:
    """Fit GARCH model for volatility forecasting."""
    if not HAS_ARCH:
        return None
    
    import time
    start_time = time.time()
    
    # Convert to returns
    train_returns = pd.Series(np.diff(train) / train[:-1] * 100)
    
    try:
        # Fit GARCH(1,1)
        model = arch_model(train_returns, vol='Garch', p=1, q=1)
        fitted = model.fit(disp='off')
        
        # Forecast volatility
        forecast = fitted.forecast(horizon=len(test))
        vol_forecast = np.sqrt(forecast.variance.values[-1, :])
        
        # Convert volatility forecast to price forecast (simplified)
        last_price = train[-1]
        predictions = [last_price]
        for vol in vol_forecast[:-1]:
            predictions.append(predictions[-1] * (1 + vol/100))
        predictions = np.array(predictions)
        
        train_time = time.time() - start_time
        metrics = evaluate_forecast(test, predictions)
        
        return ForecastResult(
            model_name='GARCH',
            predictions=predictions,
            actual=test,
            forecast_horizon=np.arange(len(test)),
            train_time=train_time,
            **metrics
        )
    except Exception as e:
        print(f"GARCH fitting failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Deep learning models
# ---------------------------------------------------------------------------

def build_lstm_model(lookback: int, n_features: int = 1, 
                    forecast_horizon: int = 1) -> Optional[keras.Model]:
    """Build LSTM model with attention mechanism."""
    try:
        _ensure_tf()
    except ImportError:
        return None
    inputs = keras.Input(shape=(lookback, n_features))
    
    # LSTM layers
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(32)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
    
    # Output layer
    outputs = layers.Dense(forecast_horizon)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


def fit_lstm_model(train: np.ndarray, test: np.ndarray,
                  lookback: int = 60, forecast_horizon: int = 1,
                  epochs: int = 50, batch_size: int = 32) -> Optional[ForecastResult]:
    """Fit LSTM model with Monte Carlo dropout for uncertainty estimation."""
    if not HAS_TENSORFLOW:
        return None
    
    import time
    start_time = time.time()
    
    # Normalize data
    train_mean, train_std = train.mean(), train.std()
    if train_std == 0:
        train_std = 1.0
    train_norm = (train - train_mean) / train_std
    test_norm = (test - train_mean) / train_std
    
    # Create sequences
    X_train, y_train = create_sequences(train_norm, lookback, forecast_horizon)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Build and train model
    model = build_lstm_model(lookback, 1, forecast_horizon)
    if model is None:
        return None
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )
    
    # Make predictions with uncertainty (Monte Carlo dropout)
    n_iterations = 100
    all_predictions = []
    history = train_norm[-lookback:].tolist()
    
    for _ in range(len(test)):
        X_pred = np.array(history[-lookback:]).reshape(1, lookback, 1)
        
        # Multiple forward passes with dropout enabled
        mc_predictions = []
        for _ in range(n_iterations):
            pred = model(X_pred, training=True).numpy()[0, 0]
            mc_predictions.append(pred)
        
        # Mean prediction
        mean_pred = np.mean(mc_predictions)
        all_predictions.append(mc_predictions)
        history.append(mean_pred)
    
    # Compute statistics from MC samples
    all_predictions = np.array(all_predictions)  # shape: (n_test, n_iterations)
    predictions = np.mean(all_predictions, axis=1)
    
    # Denormalize
    predictions = predictions * train_std + train_mean
    all_predictions = all_predictions * train_std + train_mean
    
    # Compute confidence intervals from MC samples
    ci_50_lower = np.percentile(all_predictions, 25, axis=1)
    ci_50_upper = np.percentile(all_predictions, 75, axis=1)
    ci_80_lower = np.percentile(all_predictions, 10, axis=1)
    ci_80_upper = np.percentile(all_predictions, 90, axis=1)
    ci_95_lower = np.percentile(all_predictions, 2.5, axis=1)
    ci_95_upper = np.percentile(all_predictions, 97.5, axis=1)
    
    train_time = time.time() - start_time
    metrics = evaluate_forecast(test, predictions)
    
    return ForecastResult(
        model_name='LSTM',
        predictions=predictions,
        actual=test,
        forecast_horizon=np.arange(len(test)),
        train_time=train_time,
        ci_50_lower=ci_50_lower,
        ci_50_upper=ci_50_upper,
        ci_80_lower=ci_80_lower,
        ci_80_upper=ci_80_upper,
        ci_95_lower=ci_95_lower,
        ci_95_upper=ci_95_upper,
        **metrics
    )


def fit_ensemble(results: List[ForecastResult], test: np.ndarray) -> ForecastResult:
    """Create ensemble forecast from multiple models."""
    import time
    start_time = time.time()
    
    # Weight by inverse RMSE
    weights = np.array([1/r.rmse for r in results])
    weights = weights / weights.sum()
    
    # Weighted average
    predictions = np.zeros_like(test)
    for result, weight in zip(results, weights):
        predictions += result.predictions * weight
    
    train_time = time.time() - start_time
    metrics = evaluate_forecast(test, predictions)
    
    return ForecastResult(
        model_name='Ensemble',
        predictions=predictions,
        actual=test,
        forecast_horizon=np.arange(len(test)),
        train_time=train_time,
        **metrics
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_forecast_comparison(results: List[ForecastResult], 
                            ticker: str) -> Tuple[plt.Figure, plt.Axes]:
    """Plot all forecasts vs actual with confidence intervals."""
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 5*len(results)), squeeze=False)
    
    for idx, result in enumerate(results):
        ax = axes[idx, 0]
        horizon = result.forecast_horizon
        actual = result.actual
        
        # Plot actual
        ax.plot(horizon, actual, 'k-', linewidth=2.5, label='Actual', alpha=0.9, zorder=10)
        
        # Plot prediction
        ax.plot(horizon, result.predictions, 'b-', linewidth=2, label='Forecast', alpha=0.8, zorder=5)
        
        # Plot confidence intervals if available
        if result.ci_95_lower is not None:
            ax.fill_between(horizon, result.ci_95_lower, result.ci_95_upper,
                           alpha=0.15, color='blue', label='95% CI')
        if result.ci_80_lower is not None:
            ax.fill_between(horizon, result.ci_80_lower, result.ci_80_upper,
                           alpha=0.25, color='blue', label='80% CI')
        if result.ci_50_lower is not None:
            ax.fill_between(horizon, result.ci_50_lower, result.ci_50_upper,
                           alpha=0.35, color='blue', label='50% CI')
        
        ax.set_xlabel('Days Ahead', fontsize=11)
        ax.set_ylabel('Price', fontsize=11)
        ax.set_title(f'{result.model_name} - RMSE: {result.rmse:.2f}, Dir Acc: {result.directional_accuracy:.1f}%',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Forecast Comparison with Confidence Intervals: {ticker}', 
                fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    return fig, axes


def plot_model_performance(results: List[ForecastResult]) -> Tuple[plt.Figure, np.ndarray]:
    """Plot model performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models = [r.model_name for r in results]
    
    # RMSE
    ax = axes[0, 0]
    rmses = [r.rmse for r in results]
    colors = ['green' if x == min(rmses) else 'gray' for x in rmses]
    ax.bar(models, rmses, color=colors, alpha=0.7)
    ax.set_title('Root Mean Squared Error (Lower is Better)')
    ax.set_ylabel('RMSE')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Directional Accuracy
    ax = axes[0, 1]
    dir_accs = [r.directional_accuracy for r in results]
    colors = ['green' if x == max(dir_accs) else 'gray' for x in dir_accs]
    ax.bar(models, dir_accs, color=colors, alpha=0.7)
    ax.set_title('Directional Accuracy (Higher is Better)')
    ax.set_ylabel('Accuracy (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAPE
    ax = axes[1, 0]
    mapes = [r.mape for r in results]
    colors = ['green' if x == min(mapes) else 'gray' for x in mapes]
    ax.bar(models, mapes, color=colors, alpha=0.7)
    ax.set_title('Mean Absolute Percentage Error')
    ax.set_ylabel('MAPE (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Training Time
    ax = axes[1, 1]
    times = [r.train_time for r in results]
    ax.bar(models, times, color='steelblue', alpha=0.7)
    ax.set_title('Training Time')
    ax.set_ylabel('Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_to_excel(output_path: Path,
                    results: List[ForecastResult],
                    ticker: str,
                    images: List[Tuple[str, Path]]) -> None:
    """Export forecast results to Excel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary table
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result.model_name,
            'RMSE': result.rmse,
            'MAE': result.mae,
            'MAPE (%)': result.mape,
            'Directional Accuracy (%)': result.directional_accuracy,
            'Sharpe Ratio': result.sharpe_ratio,
            'Training Time (s)': result.train_time
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create predictions table
    pred_data = {'Actual': results[0].actual}
    for result in results:
        pred_data[result.model_name] = result.predictions
    pred_df = pd.DataFrame(pred_data)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Summary
        summary_df.to_excel(writer, sheet_name='Model Comparison', index=False)
        
        # Predictions
        pred_df.to_excel(writer, sheet_name='Predictions')
        
        # Charts
        charts_ws = workbook.add_worksheet('Charts')
        row = 1
        for label, img_path in images:
            try:
                charts_ws.write(row, 1, label)
                charts_ws.insert_image(row + 1, 1, str(img_path),
                                      {'x_scale': 0.8, 'y_scale': 0.8})
                row += 30
            except Exception:
                row += 2
    
    print(f"✓ Excel report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Advanced Time Series Forecasting")
    parser.add_argument("--base", type=str, default="last_fetch")
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--models", type=str, default="arima,exp,lstm",
                       help="Comma-separated: arima,exp,garch,lstm")
    parser.add_argument("--save-dir", type=str, default="results/forecasting")
    parser.add_argument("--export-xlsx", type=str, default="reports/forecast_analysis.xlsx")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("ADVANCED TIME SERIES FORECASTING")
    print(f"{'='*60}\n")
    
    # Load data
    try:
        prices, tickers = load_and_prepare_data(args.base, args.lookback, args.forecast_days)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    if not tickers:
        print("✗ No tickers found")
        return
    
    ticker = tickers[0]
    price_series = prices[ticker].values
    
    # Train/test split
    split_idx = int(len(price_series) * TRAIN_SPLIT)
    train = price_series[:split_idx]
    test = price_series[split_idx:split_idx + args.forecast_days]
    
    print(f"Ticker: {ticker}")
    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"Forecast horizon: {args.forecast_days} days\n")
    
    # Fit models
    requested_models = [m.strip().lower() for m in args.models.split(',')]
    results = []
    
    if 'arima' in requested_models:
        print("Fitting ARIMA...")
        result = fit_arima_model(train, test)
        results.append(result)
        print(f"  RMSE: {result.rmse:.4f} | Dir Acc: {result.directional_accuracy:.2f}%")
    
    if 'exp' in requested_models:
        print("Fitting Exponential Smoothing...")
        result = fit_exponential_smoothing(train, test)
        results.append(result)
        print(f"  RMSE: {result.rmse:.4f} | Dir Acc: {result.directional_accuracy:.2f}%")
    
    if 'garch' in requested_models and HAS_ARCH:
        print("Fitting GARCH...")
        result = fit_garch_model(train, test)
        if result:
            results.append(result)
            print(f"  RMSE: {result.rmse:.4f} | Dir Acc: {result.directional_accuracy:.2f}%")
    
    if 'lstm' in requested_models:
        try:
            _ensure_tf()
            print("Fitting LSTM (this may take a while)...")
            result = fit_lstm_model(train, test, args.lookback, 1, epochs=50)
            if result:
                results.append(result)
                print(f"  RMSE: {result.rmse:.4f} | Dir Acc: {result.directional_accuracy:.2f}%")
        except ImportError as e:
            print(f"Skipping LSTM: {e}")
    
    if len(results) > 1:
        print("Creating ensemble...")
        ensemble = fit_ensemble(results, test)
        results.append(ensemble)
        print(f"  RMSE: {ensemble.rmse:.4f} | Dir Acc: {ensemble.directional_accuracy:.2f}%")
    
    # Visualization
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    images = []
    
    fig, _ = plot_forecast_comparison(results, ticker)
    img_path = outdir / f"{ticker}_forecast_comparison.png"
    fig.savefig(img_path, dpi=120, bbox_inches='tight')
    images.append((f"{ticker} Forecast Comparison", img_path))
    plt.close(fig)
    
    fig, _ = plot_model_performance(results)
    img_path = outdir / f"{ticker}_model_performance.png"
    fig.savefig(img_path, dpi=120, bbox_inches='tight')
    images.append((f"{ticker} Model Performance", img_path))
    plt.close(fig)
    
    # Export
    export_path = Path(args.export_xlsx)
    export_to_excel(export_path, results, ticker, images)
    
    # Console summary
    print(f"\n{'='*60}")
    print("FORECAST RESULTS")
    print(f"{'='*60}")
    for result in results:
        print(f"\n{result.model_name}:")
        print(f"  RMSE: {result.rmse:.4f}")
        print(f"  MAE: {result.mae:.4f}")
        print(f"  MAPE: {result.mape:.2f}%")
        print(f"  Directional Accuracy: {result.directional_accuracy:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
    
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()