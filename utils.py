"""Utility functions for the Streamlit stock forecasting app.

This module is written to be beginner-friendly:
- Clear function/class docstrings explain inputs/outputs and key steps.
- Inline comments highlight why each step is needed.
- Return types use standard numpy/pandas/Python types Streamlit can display easily.

Tip: TensorFlow/Keras is optional. If it's not installed, use the fast scikit-learn model.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow is optional at import time. We only require it if a neural model is selected.
try:
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - allow running without TF installed
    keras = None  # type: ignore
    layers = None  # type: ignore

"""Sentiment-related utilities removed by request."""


@dataclass
class ModelConfig:
    """Configuration for the time-series model.

    Attributes
    - window: Number of past days used to predict the next day (sliding window size).
    - horizon: How many days to forecast into the future.
    - test_split: Fraction of data reserved for testing (chronological split).
    - filter_outliers: Drop days with very large percent moves (see outlier_threshold).
    - outlier_threshold: Inclusive percent threshold for filtering (e.g., 5.0 = ±5%).
    - soften_spikes: Smooth unusually large moves in the training region.
    - spike_threshold: Inclusive percent threshold for spike softening.
    - spike_factor: Softening factor: new = prev + factor * (curr - prev).
    """
    window: int = 30
    horizon: int = 10
    test_split: float = 0.1
    filter_outliers: bool = False
    outlier_threshold: float = 5.0  # percent, inclusive
    soften_spikes: bool = False
    spike_threshold: float = 10.0  # percent, inclusive
    spike_factor: float = 0.5      # new = prev + factor * delta
    # Optional additional features (low-risk): include pct_change and rolling mean
    add_technical_features: bool = False
    rolling_window: int = 7
    # Optional rolling mean feature (separate from legacy add_technical_features)
    add_rolling_mean: bool = False
    # Optional specific technical indicators
    # SMA window: set >0 to enable a single SMA feature (window length in days)
    sma_window: int = 0
    add_rsi: bool = False
    rsi_window: int = 14
    # Scaler choice: 'standard', 'minmax', 'robust', 'log', 'diff'
    scaler: str = "standard"


def fetch_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLC data and return a minimal DataFrame.

    Parameters
    - ticker: Symbol like "AAPL" or exchange-qualified like "RELIANCE.NS".
    - period: Range string understood by yfinance (e.g., "1mo", "1y", "5y", "10y", "max").

    Returns
    - DataFrame with columns [date (timezone-naive), open, price]; the 'date' column is always timezone-naive after processing.

    Raises
    - RuntimeError if no data is returned.
    """
    # Fetch price history for a symbol using yfinance.
    # Returns a minimal DataFrame with timezone-naive dates and columns [date, open, price].
    # Normalizes period codes and applies optional long-window cutoffs (20y/25y).
    t = yf.Ticker(ticker)
    code = (period or "1y").lower()
    # Map extended codes; yfinance doesn't support 20y/25y directly, so filter from max
    if code in {"1w", "1week"}:
        start = (dt.datetime.now(dt.timezone.utc) -
                 dt.timedelta(days=7)).date()
        df = t.history(start=start, interval="1d", auto_adjust=False)
    elif code in {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd"}:
        df = t.history(period=code, interval="1d", auto_adjust=False)
    elif code in {"20y", "25y"}:
        df = t.history(period="max", interval="1d", auto_adjust=False)
        years = 20 if code == "20y" else 25
        cutoff = (dt.datetime.now(dt.timezone.utc) -
                  dt.timedelta(days=365 * years)).date()
        # Will filter after reset_index/rename
    elif code in {"max"}:
        df = t.history(period="max", interval="1d", auto_adjust=False)
    else:
        # Fallback
        df = t.history(period="1y", interval="1d", auto_adjust=False)
    if df.empty:
        raise RuntimeError("No data from Yahoo Finance")
    df = df.reset_index()
    df.rename(columns={"Date": "date", "Close": "price",
              "Open": "open"}, inplace=True)
    df = df[["date", "open", "price"]].dropna()
    # Ensure timezone-naive datetimes for safe comparisons
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    # Apply 20y/25y cutoff if needed
    if code in {"20y", "25y"}:
        years = 20 if code == "20y" else 25
        cutoff = pd.Timestamp.now(tz="UTC").tz_convert(
            None) - pd.DateOffset(years=years)
        df = df[df["date"] >= cutoff]
    return df


def search_tickers(query: str, limit: int = 10, region: str = "US", lang: str = "en-US") -> List[Dict[str, Any]]:
    """Search Yahoo Finance for symbols matching a query.

    Returns a list of dictionaries with: {symbol, name, type, exch, exchDisp}.
    """
    # Perform a best-effort symbol search against Yahoo Finance's search endpoint.
    # Returns a list of small dicts: {symbol, name, type, exch, exchDisp} or [] on failure.
    # Uses a short timeout and a browser-like User-Agent to reduce trivial blocks.
    q = (query or "").strip()
    if len(q) < 2:
        return []
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        "q": q,
        "quotesCount": max(1, int(limit)),
        "newsCount": 0,
        "listsCount": 0,
        "lang": lang,
        "region": region,
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    import json
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json() or {}
        quotes = data.get("quotes", []) or []
        out: List[Dict[str, Any]] = []
        for it in quotes:
            symbol = it.get("symbol")
            name = it.get("shortname") or it.get(
                "longname") or it.get("quoteType") or ""
            qt = (it.get("quoteType") or "").upper()
            exch = it.get("exch") or it.get("exchange") or ""
            exchDisp = it.get("exchDisp") or it.get("exchangeDisp") or exch
            if not symbol:
                continue
            # Prefer common types
            if qt and qt not in {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}:
                # still include but deprioritize; we'll add anyway and let caller limit
                pass
            out.append({
                "symbol": symbol,
                "name": name,
                "type": qt,
                "exch": exch,
                "exchDisp": exchDisp,
            })
        return out[:limit]
    except requests.RequestException:
        return []
    except json.JSONDecodeError:
        return []


def validate_ticker_symbol(symbol: str) -> bool:
    """Return True if the symbol has any daily bars in the last month."""
    # Quick health-check for a ticker symbol: try fetching 1 month of daily bars.
    # Returns True when yfinance returns any rows; False on empty data or exceptions.
    # Useful to validate user input before attempting heavier operations.
    s = (symbol or "").strip().upper()
    if not s:
        return False
    try:
        t = yf.Ticker(s)
        df = t.history(period="1mo", interval="1d", auto_adjust=False)
        return not df.empty
    except Exception:
        # yfinance's internal exception classes vary across versions; fallback to a broad catch
        return False


def inclusive_outlier_mask(series: pd.Series, pct_threshold: float) -> pd.Series:
    """Boolean mask (True=keep) for rows whose |pct_change| <= threshold.

    The first row is always kept (pct_change is NaN).
    """
    # Compute percent change and return a boolean mask that keeps values
    # whose absolute day-over-day percent change is within the threshold.
    # The first row is always kept because pct_change is NaN there.
    pct = series.pct_change() * 100.0
    mask = pct.abs().le(pct_threshold) | pct.isna()  # keep first NaN
    # Always keep the first value
    mask.iloc[0] = True
    return mask


def soften_spikes_train_only(series: pd.Series, train_end_idx: int, threshold: float, factor: float) -> Tuple[pd.Series, List[int]]:
    """Soften unusually large moves in the training region only.

    For indices 1..train_end_idx, when |pct_change| >= threshold, pull the value toward
    the previous one by a fraction (factor). Returns the modified series and changed indices.
    """
    # Copy the series and reduce extreme day-over-day moves inside the train region.
    # When a pct change exceeds the threshold, pull the value toward the previous
    # value by the provided factor. Returns the modified series and changed indices.
    s = series.copy().astype(float)
    changed_idx: List[int] = []
    pct = s.pct_change() * 100.0
    for i in range(1, min(len(s), train_end_idx + 1)):
        if abs(pct.iloc[i]) >= threshold:
            prev = s.iloc[i - 1]
            delta = s.iloc[i] - prev
            s.iloc[i] = prev + factor * delta
            changed_idx.append(i)
    return s, changed_idx


def make_windows_close_only(values: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows (X) and multi-step targets (y) from a 1D array.

    Returns X shaped (n_windows, window) and y shaped (n_windows, horizon).
    When horizon==1 the second dimension is 1 (kept as 2D for consistency).
    """
    X, y = [], []
    n = len(values)
    # Last valid start index for a window is n - horizon
    for i in range(window, n - horizon + 1):
        X.append(values[i - window: i])
        # targets are the next `horizon` values
        ys = [values[i + k] for k in range(horizon)]
        y.append(ys)
    return np.array(X), np.array(y)


def make_windows_multifeature(values: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for multifeature inputs.

    values: 2D array shaped (n_samples, n_features)
    Returns X of shape (n_windows, window, n_features) and y of shape (n_windows, horizon)
    where y for step k is the next-step value of the first column (close) at offset k.
    """
    if values.ndim == 1:
        # fallback to 1D handler
        return make_windows_close_only(values, window, horizon=horizon)
    n_rows, n_feats = values.shape
    X = []
    y = []
    # i is the index of the first target (i.e., window end)
    for i in range(window, n_rows - horizon + 1):
        X.append(values[i - window: i])
        ys = [values[i + k, 0] for k in range(horizon)]
        y.append(ys)
    return np.array(X), np.array(y)


def _compile(seq: Any, learning_rate: float = 1e-3) -> Any:
    """Compile a Keras model using Adam optimizer and MSE loss."""
    # Helper to compile a Keras model consistently with Adam and MSE.
    # Keeps model creation code concise and centralizes optimizer settings.
    # Helper to compile with a consistent optimizer/loss
    seq.compile(optimizer=keras.optimizers.Adam(
        learning_rate=learning_rate), loss="mse")
    return seq


def _build_keras_model_lstm(input_len: int, dropout: float = 0.3, learning_rate: float = 1e-3, output_units: int = 1) -> Any:
    """Simple LSTM regressor for univariate sequences (shape: [steps=input_len, 1])."""
    # Build a compact LSTM-based regressor for single-feature sequences.
    # Returns a compiled Keras Sequential model ready to fit.
    if keras is None or layers is None:
        raise RuntimeError(
            "Neural models require TensorFlow/Keras. Select 'sklearn (fast)' or install a compatible TensorFlow package."
        )
    seq = keras.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(output_units),
    ])
    return _compile(seq, learning_rate)


def _build_keras_model_gru(input_len: int, dropout: float = 0.3, learning_rate: float = 1e-3, output_units: int = 1) -> Any:
    """Simple GRU regressor for univariate sequences."""
    # Build a lightweight GRU regressor as an alternative to the LSTM.
    # Useful when GRU's gating gives similar performance with fewer parameters.
    if keras is None or layers is None:
        raise RuntimeError(
            "Neural models require TensorFlow/Keras. Select 'sklearn (fast)' or install a compatible TensorFlow package."
        )
    seq = keras.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.GRU(64, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(output_units),
    ])
    return _compile(seq, learning_rate)


def _build_keras_model_bilstm(input_len: int, dropout: float = 0.3, learning_rate: float = 1e-3, output_units: int = 1) -> Any:
    """Bidirectional LSTM regressor."""
    # Create a bidirectional LSTM to allow the model to see sequence context
    # in both directions; still used for univariate regression tasks here.
    if keras is None or layers is None:
        raise RuntimeError(
            "Neural models require TensorFlow/Keras. Select 'sklearn (fast)' or install a compatible TensorFlow package."
        )
    seq = keras.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(dropout),
        layers.Dense(output_units),
    ])
    return _compile(seq, learning_rate)


def _build_keras_model_bigru(input_len: int, dropout: float = 0.3, learning_rate: float = 1e-3, output_units: int = 1) -> Any:
    """Bidirectional GRU regressor."""
    # Bidirectional GRU variant; trades architectural differences against training cost.
    # Returns a compiled model with dropout for regularization.
    if keras is None or layers is None:
        raise RuntimeError(
            "Neural models require TensorFlow/Keras. Select 'sklearn (fast)' or install a compatible TensorFlow package."
        )
    seq = keras.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Bidirectional(layers.GRU(64, return_sequences=False)),
        layers.Dropout(dropout),
        layers.Dense(output_units),
    ])
    return _compile(seq, learning_rate)


def _build_keras_model_deep_lstm(input_len: int, dropout: float = 0.3, learning_rate: float = 1e-3, output_units: int = 1) -> Any:
    """Two-layer (stacked) LSTM regressor."""
    # Stacked LSTM architecture (deeper) for capturing more complex temporal patterns.
    # Use with care on small datasets to avoid overfitting.
    if keras is None or layers is None:
        raise RuntimeError(
            "Neural models require TensorFlow/Keras. Select 'sklearn (fast)' or install a compatible TensorFlow package."
        )
    seq = keras.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(output_units),
    ])
    return _compile(seq, learning_rate)


def _build_keras_model_deep_transition_lstm(input_len: int, dropout: float = 0.3, learning_rate: float = 1e-3, output_units: int = 1) -> Any:
    """Deep-transition LSTM: multiple LSTMCells stacked inside a single RNN cell.

    This increases per-time-step transition depth (richer state-update) while
    keeping the number of unrolled recurrent layers small. Useful when the
    per-step dynamics are complex but you prefer fewer stacked recurrent layers.
    """
    if keras is None or layers is None:
        raise RuntimeError(
            "Neural models require TensorFlow/Keras. Select 'sklearn (fast)' or install a compatible TensorFlow package."
        )
    # Build a StackedRNNCells composed of multiple LSTMCell units to create
    # a deeper transition for each time-step.
    try:
        # Layers API: StackedRNNCells lives under layers
        cells = [layers.LSTMCell(64), layers.LSTMCell(64)]
        stacked = layers.StackedRNNCells(cells)
        r = layers.RNN(stacked, return_sequences=False)
        seq = keras.Sequential([
            layers.Input(shape=(input_len, 1)),
            r,
            layers.Dropout(dropout),
            layers.Dense(output_units),
        ])
    except Exception:
        # Fallback: two-layer stacked LSTM (behaves similarly at a high level)
        seq = keras.Sequential([
            layers.Input(shape=(input_len, 1)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(dropout),
            layers.Dense(output_units),
        ])
    return _compile(seq, learning_rate)


def _build_keras_model_deep_gru(input_len: int, dropout: float = 0.3, learning_rate: float = 1e-3, output_units: int = 1) -> Any:
    """Two-layer (stacked) GRU regressor."""
    # Two-layer GRU network for deeper sequence modeling without LSTM complexity.
    # Compiled and returned ready for training.
    if keras is None or layers is None:
        raise RuntimeError(
            "Neural models require TensorFlow/Keras. Select 'sklearn (fast)' or install a compatible TensorFlow package."
        )
    seq = keras.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(dropout),
        layers.GRU(32, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(output_units),
    ])
    return _compile(seq, learning_rate)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: RMSE, MAE, MAPE (%), and R²."""
    # Compute and return common regression metrics used for evaluation.
    # Handles small/empty inputs safely by returning NaNs where appropriate.
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan"), "R2": float("nan")}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def _train_and_forecast_core(
    df: pd.DataFrame,
    cfg: ModelConfig,
    model_kind: str,
    epochs: int,
    batch_size: int,
    dropout: float,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train a model on close prices and forecast future values.

    Pipeline overview
    1) Optional: filter outliers and/or soften spikes (train only)
    2) Scale prices (fit on train subset only to avoid leakage)
    3) Build sliding windows on the scaled series
    4) Chronologically split into train and test
    5) Train either a sklearn regressor or a Keras model
    6) Evaluate on test (inverse-scaled metrics)
    7) Forecast `horizon` future days iteratively (capped to 7)
    """
    # Central pipeline: prepare data, scale, window, split, train and forecast.
    # Supports both scikit-learn (fast) and several Keras model kinds, returning
    # a result dict with fitted series, metrics, future forecasts and diagnostic data.
    # Prepare base series
    close = df["price"].astype(float).reset_index(drop=True)

    # Outlier filtering on absolute day-over-day % change (inclusive)
    kept_mask = pd.Series([True] * len(close))
    if cfg.filter_outliers:
        mask = inclusive_outlier_mask(close, cfg.outlier_threshold)
        kept_mask = mask
        close_filtered = close[mask].reset_index(drop=True)
    else:
        close_filtered = close.copy()

    # Train/test split (chronological)
    n = len(close_filtered)
    test_n = max(1, int(n * cfg.test_split))
    train_n = max(1, n - test_n)

    # Train-only spike softening before scaling
    close_train = close_filtered.iloc[:train_n]

    changed_idx: List[int] = []
    if cfg.soften_spikes:
        soft_train, changed = soften_spikes_train_only(
            close_filtered, train_end_idx=train_n - 1, threshold=cfg.spike_threshold, factor=cfg.spike_factor
        )
        close_train = soft_train.iloc[:train_n]
        changed_idx = [i for i in changed if i < train_n]

    # Scale (fit on training portion only)
    # Optional: add technical features (pct_change, rolling mean) to provide extra signals.
    if cfg.add_technical_features:
        # Build a DataFrame of features aligned to the filtered series
        # Apply optional preprocessing transforms (log / difference) to the base close series
        close_proc = close_filtered.copy().astype(float)
        if cfg.scaler == "log":
            close_proc = np.log1p(close_proc)
        elif cfg.scaler == "diff":
            close_proc = close_proc.diff().fillna(0)

        df_feats = pd.DataFrame({"close": close_proc.values})
        # Rolling mean (on raw close) with window cfg.rolling_window; align center=False
        # Keep backwards compatibility: if legacy `add_technical_features` is True,
        # still include rolling_mean by default. Otherwise include only when
        # cfg.add_rolling_mean is enabled.
        rw = max(1, int(cfg.rolling_window))
        if getattr(cfg, "add_rolling_mean", False) or cfg.add_technical_features:
            df_feats["rolling_mean"] = pd.Series(close_proc).rolling(
                window=rw, min_periods=1).mean().values

        # Simple Moving Averages (SMA) - optional
        if getattr(cfg, "sma_window", 0) > 0:
            w = int(cfg.sma_window)
            df_feats["sma"] = pd.Series(close_proc).rolling(
                window=w, min_periods=1).mean().values

        # RSI (relative strength index) - optional, classic 14-day default
        if getattr(cfg, "add_rsi", False):
            # Calculate RSI using the Wilder smoothing approximation
            window_rsi = max(2, int(getattr(cfg, "rsi_window", 14)))
            delta = pd.Series(close_proc).diff().fillna(0)
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            # Wilder smoothing: first value is simple average, then exponential-like
            roll_up = up.ewm(alpha=1.0/window_rsi, adjust=False).mean()
            roll_down = down.ewm(alpha=1.0/window_rsi, adjust=False).mean()
            rs = roll_up / (roll_down.replace(0, np.nan))
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(0)
            df_feats["rsi"] = rsi.values

        # Train scalers across columns: fit on training rows only
        # Choose scaler implementation
        scaler_name = (cfg.scaler or "standard").lower()
        if scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            # For 'standard', 'log', 'diff' and unknown values, use StandardScaler
            scaler = StandardScaler()

        # Fit scaler on training rows only
        train_feat_vals = df_feats.iloc[:train_n].values
        scaler.fit(train_feat_vals)
        full_scaled = scaler.transform(df_feats.values)

        # Persist metadata for inverse transforms
        fitted_scaler = scaler
        fitted_feature_cols = df_feats.columns.tolist()

        # full_scaled is 2D: (n_samples, n_features)
        # Create windows for multifeature inputs (respect the configured horizon)
        X_all, y_all = make_windows_multifeature(
            full_scaled, cfg.window, horizon=cfg.horizon)
    else:
        # Non-technical single-feature path: allow scaler choices and optional preprocess
        close_proc = close_filtered.copy().astype(float)
        if cfg.scaler == "log":
            close_proc = np.log1p(close_proc)
        elif cfg.scaler == "diff":
            close_proc = close_proc.diff().fillna(0)

        scaler_name = (cfg.scaler or "standard").lower()
        if scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        train_vals = close_proc.iloc[:train_n].values.reshape(-1, 1)
        scaler.fit(train_vals)
        full_scaled = scaler.transform(
            close_proc.values.reshape(-1, 1)).flatten()
        # Persist metadata for inverse transforms (single feature)
        fitted_scaler = scaler
        fitted_feature_cols = [close_proc.name if hasattr(
            close_proc, 'name') and close_proc.name else 'price']

    # If windows not already created (non-technical path), create them now
    if 'X_all' not in locals():
        X_all, y_all = make_windows_close_only(
            full_scaled, cfg.window, horizon=cfg.horizon)
    # Determine split index in windowed space
    split_idx = max(1, train_n - cfg.window)
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test, y_test = X_all[split_idx:], y_all[split_idx:]

    history = None
    mk = (model_kind or "sklearn").lower()
    if mk != "sklearn":
        if keras is None:
            # Fail fast with a clear message if user selects a neural model without TF available
            raise RuntimeError(
                "TensorFlow/Keras is not available in this environment. Install a TensorFlow version compatible with your Python, or switch to 'sklearn (fast)'."
            )
        # Reshape to [samples, timesteps, features] for Keras layers
        # If X_all is 3D already (multifeature), keep; if 2D (single feature), add last axis
        if X_all.ndim == 3:
            Xtr = X_train
            Xte = X_test
            Xall = X_all
        else:
            Xtr = X_train[..., None]
            Xte = X_test[..., None]
            Xall = X_all[..., None]
        builders = {
            "lstm": _build_keras_model_lstm,
            "gru": _build_keras_model_gru,
            "bilstm": _build_keras_model_bilstm,
            "deep-lstm": _build_keras_model_deep_lstm,
            "deep-gru": _build_keras_model_deep_gru,
            # alias for clarity
            "stacked-lstm": _build_keras_model_deep_lstm,
            "deep-transition-lstm": _build_keras_model_deep_transition_lstm,
        }
        builder = builders.get(mk)
        if builder is None:
            raise ValueError(f"Unknown neural model kind: {mk}")
        # Build the neural model with the desired learning rate
        # If we have multifeature windows, build the model to accept that many features
        input_feats = Xall.shape[2] if Xall.ndim == 3 else 1
        output_units = int(max(1, min(int(cfg.horizon), 14)))
        if input_feats == 1:
            model = builder(cfg.window, dropout=dropout,
                            learning_rate=float(learning_rate), output_units=output_units)
        else:
            # Dynamically construct models that mirror the simple builders
            if mk == "lstm":
                seq = keras.Sequential([
                    layers.Input(shape=(cfg.window, input_feats)),
                    layers.LSTM(64, return_sequences=False),
                    layers.Dropout(dropout),
                    layers.Dense(output_units),
                ])
            elif mk == "gru":
                seq = keras.Sequential([
                    layers.Input(shape=(cfg.window, input_feats)),
                    layers.GRU(64, return_sequences=False),
                    layers.Dropout(dropout),
                    layers.Dense(output_units),
                ])
            elif mk == "bilstm":
                seq = keras.Sequential([
                    layers.Input(shape=(cfg.window, input_feats)),
                    layers.Bidirectional(layers.LSTM(
                        64, return_sequences=False)),
                    layers.Dropout(dropout),
                    layers.Dense(output_units),
                ])
            elif mk == "bigru":
                seq = keras.Sequential([
                    layers.Input(shape=(cfg.window, input_feats)),
                    layers.Bidirectional(layers.GRU(
                        64, return_sequences=False)),
                    layers.Dropout(dropout),
                    layers.Dense(output_units),
                ])
            elif mk == "deep-lstm":
                seq = keras.Sequential([
                    layers.Input(shape=(cfg.window, input_feats)),
                    layers.LSTM(64, return_sequences=True),
                    layers.Dropout(dropout),
                    layers.LSTM(32, return_sequences=False),
                    layers.Dropout(dropout),
                    layers.Dense(output_units),
                ])
            elif mk == "stacked-lstm":
                seq = keras.Sequential([
                    layers.Input(shape=(cfg.window, input_feats)),
                    layers.LSTM(64, return_sequences=True),
                    layers.Dropout(dropout),
                    layers.LSTM(32, return_sequences=False),
                    layers.Dropout(dropout),
                    layers.Dense(output_units),
                ])
            elif mk == "deep-gru":
                seq = keras.Sequential([
                    layers.Input(shape=(cfg.window, input_feats)),
                    layers.GRU(64, return_sequences=True),
                    layers.Dropout(dropout),
                    layers.GRU(32, return_sequences=False),
                    layers.Dropout(dropout),
                    layers.Dense(output_units),
                ])
            else:
                # Fallback to single-feature builder if unknown
                model = builder(cfg.window, dropout=dropout,
                                learning_rate=float(learning_rate))
                seq = None
            if seq is not None:
                model = _compile(seq, learning_rate=float(learning_rate))
        # Keras training with optional callbacks
        total_epochs = int(max(1, epochs))
        history = {"loss": [], "val_loss": []}
        # Allow optional callbacks: EarlyStopping and ModelCheckpoint
        keras_callbacks = []
        if callbacks is None:
            callbacks = {}
        if callbacks.get("early_stopping", True):
            try:
                keras_callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True)
                )
            except Exception:
                pass
        if callbacks.get("model_checkpoint", True):
            try:
                # store checkpoint in a temporary path
                import tempfile
                import os
                ckpt_dir = tempfile.gettempdir()
                ckpt_path = os.path.join(
                    ckpt_dir, f"model_ckpt_{np.random.randint(1e9)}.h5")
                keras_callbacks.append(
                    keras.callbacks.ModelCheckpoint(
                        ckpt_path, save_best_only=True, monitor="val_loss")
                )
            except Exception:
                pass

        # Simple in-memory model cache to avoid retraining identical configs during iteration
        # Key by a tuple of (model_kind, window, horizon, input_feats, learning_rate, epochs)
        try:
            if "_model_cache" not in globals():
                _model_cache = {}
            cache_key = (mk, cfg.window, cfg.horizon, int(
                input_feats), float(learning_rate), int(total_epochs))
            cached = _model_cache.get(cache_key)
            if cached is not None and not callbacks.get("force_retrain", False):
                model = cached
            else:
                # Train one epoch at a time to report progress
                for ep in range(total_epochs):
                    h = model.fit(
                        Xtr,
                        y_train,
                        validation_data=(Xte, y_test) if len(Xte) else None,
                        epochs=1,
                        batch_size=int(max(1, batch_size)),
                        verbose=0,
                        callbacks=keras_callbacks if len(
                            keras_callbacks) else None,
                    )
                    l = float(h.history.get("loss", [np.nan])[-1])
                    vl = float(h.history.get(
                        "val_loss", [np.nan])[-1]) if len(Xte) else np.nan
                    history["loss"].append(l)
                    if not np.isnan(vl):
                        history["val_loss"].append(vl)
                    if callable(progress_callback):
                        try:
                            progress_callback(ep + 1, total_epochs, l)
                        except Exception:
                            pass
                # store trained model in cache
                try:
                    _model_cache[cache_key] = model
                except Exception:
                    pass
        except Exception:
            # Fallback: plain training loop if cache machinery fails
            for ep in range(total_epochs):
                h = model.fit(
                    Xtr,
                    y_train,
                    validation_data=(Xte, y_test) if len(Xte) else None,
                    epochs=1,
                    batch_size=int(max(1, batch_size)),
                    verbose=0,
                )
                l = float(h.history.get("loss", [np.nan])[-1])
                vl = float(h.history.get(
                    "val_loss", [np.nan])[-1]) if len(Xte) else np.nan
                history["loss"].append(l)
                if not np.isnan(vl):
                    history["val_loss"].append(vl)
                if callable(progress_callback):
                    try:
                        progress_callback(ep + 1, total_epochs, l)
                    except Exception:
                        pass
        # Predict fitted (train+test) multi-step outputs: shape -> (n_windows, horizon)
        y_pred_all = model.predict(Xall, verbose=0)
    else:
        # Fast, CPU-friendly regressor
        # For multi-output regression with sklearn wrap a single-output regressor
        from sklearn.multioutput import MultiOutputRegressor
        base = GradientBoostingRegressor(random_state=42)
        model = MultiOutputRegressor(base)
        # sklearn expects 2D inputs: flatten the window dimension and features
        if X_all.ndim == 3:
            n_samples, w, n_feats = X_all.shape
            X_all_flat = X_all.reshape(n_samples, w * n_feats)
            X_train_flat = X_train.reshape(
                X_train.shape[0], X_train.shape[1] * X_train.shape[2])
            X_test_flat = X_test.reshape(
                X_test.shape[0], X_test.shape[1] * X_test.shape[2])
        else:
            X_all_flat = X_all
            X_train_flat = X_train
            X_test_flat = X_test
        # y_train is shape (n_samples, horizon) - MultiOutputRegressor accepts that
        model.fit(X_train_flat, y_train)
        # Fitted (train+test) backcast: shape (n_windows, horizon)
        y_pred_all = model.predict(X_all_flat)

    # Metrics on test
    def inv(v):
        """Inverse transform a 1D array of predicted/scaled close values.

        If the scaler was fit on multiple features, construct a placeholder 2D
        array where the first column is `v` and remaining columns are the
        corresponding last-observed values (so inverse_transform can run).
        Return the first column of the inverse-transformed array.
        """
        arr = np.array(v).reshape(-1, 1)
        # Use persisted scaler and feature_cols when present
        sc = locals().get('fitted_scaler', None)
        fcols = locals().get('fitted_feature_cols', None)
        if sc is not None:
            scaler_local = sc
        else:
            scaler_local = scaler
        # determine n_in from the scaler we will actually use
        n_in = getattr(scaler_local, 'n_features_in_', None)
        if n_in is None or n_in == 1:
            return scaler_local.inverse_transform(arr).flatten()
        # sklearn scalers have attribute `n_features_in_` when fit
        n_in = getattr(scaler, "n_features_in_", None)
        if n_in is None or n_in == 1:
            return scaler.inverse_transform(arr).flatten()
        # Build a 2D placeholder using the last available row(s) from full_scaled
        # Prefer using the last observed feature row when available; otherwise
        # fall back to the scaler's learned mean if present, then zeros.
        if isinstance(full_scaled, np.ndarray) and full_scaled.ndim == 2 and full_scaled.shape[1] == n_in:
            template = full_scaled[-1]
        elif hasattr(scaler, "mean_") and getattr(scaler, "mean_") is not None:
            template = np.array(getattr(scaler, "mean_"), dtype=float)
        else:
            template = np.zeros(int(n_in), dtype=float)

        # Create the matrix and set the first column to our predicted values
        M = np.tile(template.reshape(1, -1), (arr.shape[0], 1)).astype(float)
        try:
            M[:, 0] = arr.flatten()
        except Exception:
            # If assignment fails due to shape mismatch, try a safer broadcast
            M = np.zeros((arr.shape[0], int(n_in)), dtype=float)
            for ii in range(arr.shape[0]):
                M[ii, 0] = float(arr.flatten()[ii])

        # Attempt inverse transform; if it fails, raise a clearer error with shapes
        try:
            inv_all = scaler.inverse_transform(M)
        except Exception as e:
            raise RuntimeError(
                f"Scaler inverse_transform failed: scaler.n_features_in_={n_in}, input shape={M.shape}, full_scaled_shape={getattr(full_scaled, 'shape', None)}; original error: {e}") from e
        return inv_all[:, 0].flatten()

    # y_pred_all shape -> (n_windows, horizon)
    # Inverse-transform predicted (scaled) values back to original price units
    fitted_prices = []
    if hasattr(y_pred_all, "ndim") and getattr(y_pred_all, "ndim") == 2:
        n_windows, h = y_pred_all.shape
        # For each horizon column, inverse-transform the column and assemble
        inv_cols = []
        for col in range(h):
            inv_col = inv(y_pred_all[:, col])
            inv_cols.append(inv_col)
        inv_matrix = np.stack(inv_cols, axis=1) if len(
            inv_cols) > 0 else np.zeros((n_windows, 0))
        for i in range(n_windows):
            fitted_prices.append(list(inv_matrix[i, :]))
    else:
        # Single-column predictions (flatten) -> inverse transform and wrap
        inv_flat = inv(np.ravel(y_pred_all))
        fitted_prices = [[float(x)] for x in np.ravel(inv_flat)]

    # Metrics on test region (inverse scaled) -- compute per-horizon metrics
    m = {}
    horizon_used = int(max(1, min(int(cfg.horizon), 14)))
    if len(y_test):
        # y_test shape (n_test_windows, horizon)
        # Ensure arrays are at least 2-D so indexing [:, h] is safe when horizon==1
        y_test_arr = np.atleast_2d(np.array(y_test))
        y_pred_test_arr = np.atleast_2d(np.array(
            y_pred_all[len(y_train): len(y_train) + len(y_test)]))
        # Determine how many horizon columns are actually available on each side
        n_test_h = y_test_arr.shape[1]
        n_pred_h = y_pred_test_arr.shape[1]
        max_h = min(horizon_used, n_test_h, n_pred_h)
        # For each horizon step compute metrics (only up to available columns)
        per_h_metrics = {}
        for h in range(max_h):
            y_t = inv(y_test_arr[:, h])
            y_p = inv(y_pred_test_arr[:, h])
            per_h_metrics[f"h{h+1}"] = _metrics(y_t, y_p)
        # Also provide averaged single-number metrics for the first horizon step when present
        m = per_h_metrics.get("h1", {"RMSE": float("nan"), "MAE": float(
            "nan"), "MAPE": float("nan"), "R2": float("nan")})
        m["per_horizon"] = per_h_metrics
    else:
        m = {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float(
            "nan"), "R2": float("nan"), "per_horizon": {}}

    # Build fitted series aligned to the filtered timeline
    # For multi-step predictions, we store per-window arrays; fitted_full will be list of lists (or scalars when horizon=1)
    fitted_full = [None] * len(full_scaled)
    # windows produce predictions starting at index cfg.window
    for i, vals in enumerate(fitted_prices, start=cfg.window):
        # vals may be scalar, list, numpy array, or other iterable
        if hasattr(vals, "tolist"):
            fitted_full[i] = vals.tolist()
        elif isinstance(vals, (list, tuple, np.ndarray)):
            fitted_full[i] = list(vals)
        else:
            try:
                fitted_full[i] = list(vals)
            except Exception:
                fitted_full[i] = [vals]

    # Future forecasting (iterative) with horizon capped at 14 days
    # full_scaled may be 1D (single feature) or 2D (n_samples, n_feats)
    last_window = full_scaled[-cfg.window:]
    # Ensure last_window is a plain Python list of rows for consistent processing
    if hasattr(last_window, "tolist"):
        last_window = last_window.tolist()
    elif isinstance(last_window, (list, tuple, np.ndarray)):
        last_window = list(last_window)
    else:
        # fallback: wrap scalar into a list
        last_window = [last_window]
    future_scaled = []
    # For multi-step forecasting we call the model once per-window to predict horizon outputs
    horizon_used = int(max(1, min(int(cfg.horizon), 14)))
    # We will produce a single multi-step forecast for the immediate future based on last_window
    # Prepare input and predict horizon values in one call
    if mk != "sklearn":
        x_in = np.array(last_window, dtype=float)
        if x_in.ndim == 2:
            x_in = x_in.reshape(1, cfg.window, x_in.shape[1])
        else:
            x_in = x_in.reshape(1, cfg.window, 1)
        pred = model.predict(x_in, verbose=0)
        # pred shape -> (1, horizon) or (1,) when horizon==1
        future_scaled = list(np.ravel(pred)[:horizon_used])
    else:
        arr = np.array(last_window, dtype=float)
        if arr.ndim == 2:
            arr_flat = arr.reshape(1, arr.shape[0] * arr.shape[1])
        else:
            arr_flat = arr.reshape(1, -1)
        pred = model.predict(arr_flat)
        future_scaled = list(np.ravel(pred)[:horizon_used])
    future_prices = inv(future_scaled)
    # Normalize future_prices to a Python list for safe downstream usage
    if hasattr(future_prices, "tolist"):
        future_prices_list = future_prices.tolist()
    elif isinstance(future_prices, (list, tuple, np.ndarray)):
        future_prices_list = list(future_prices)
    else:
        future_prices_list = [future_prices]

    # --- Gap filling: optionally forecast missing calendar days up to today ---
    # Compute last original index corresponding to filtered data
    try:
        kept_indices = np.where(kept_mask.values)[0].tolist()
        last_orig_idx = kept_indices[-1] if len(kept_indices) else len(df) - 1
    except Exception:
        last_orig_idx = len(df) - 1
    try:
        last_date = pd.to_datetime(df["date"]).dt.tz_localize(
            None).iloc[last_orig_idx]
    except Exception:
        last_date = pd.to_datetime(df["date"]).dt.tz_localize(None).iloc[-1]
    # How many calendar days are missing between last_date and today
    today = pd.Timestamp.now(tz=None).normalize()
    last_norm = pd.to_datetime(last_date).normalize()
    gap_days = int(max(0, (today - last_norm).days))
    # Cap fill to at most 14 days to avoid long iterative runs
    fill_days = min(gap_days, 14)
    gap_scaled_preds: List[float] = []
    if fill_days > 0:
        # Prepare a mutable sliding window in scaled space
        # last_window currently was built from full_scaled earlier
        lw = full_scaled[-cfg.window:]
        # If single-feature scaled, ensure it's a 1D list
        single_feat = (isinstance(full_scaled, np.ndarray) and full_scaled.ndim == 1) or (
            isinstance(full_scaled, list) and len(np.array(full_scaled).shape) == 1)
        for _ in range(fill_days):
            # Build input for model depending on model kind and feature dims
            if mk != "sklearn":
                x_in = np.array(lw, dtype=float)
                if x_in.ndim == 2:
                    x_in = x_in.reshape(1, cfg.window, x_in.shape[1])
                else:
                    x_in = x_in.reshape(1, cfg.window, 1)
                pred = model.predict(x_in, verbose=0)
            else:
                arr = np.array(lw, dtype=float)
                if arr.ndim == 2:
                    arr_flat = arr.reshape(1, arr.shape[0] * arr.shape[1])
                else:
                    arr_flat = arr.reshape(1, -1)
                pred = model.predict(arr_flat)
            # Take first step's scaled prediction (first column / first element)
            p_scaled = float(np.ravel(pred)[0])
            gap_scaled_preds.append(p_scaled)
            # Slide the window: append predicted scaled value
            if single_feat:
                lw = list(lw[1:]) + [p_scaled]
            else:
                # For multifeature windows, copy last row and replace first column with p_scaled
                last_row = np.array(lw[-1], dtype=float).copy()
                last_row[0] = p_scaled
                lw = list(lw[1:]) + [last_row]
        # Inverse-transform gap predictions (scaled -> prices)
        try:
            gap_prices = inv(np.array(gap_scaled_preds).reshape(-1, 1)
                             ) if len(gap_scaled_preds) else []
        except Exception:
            # If scaler expects multiple features, call inv on 1D list
            gap_prices = inv(np.array(gap_scaled_preds)) if len(
                gap_scaled_preds) else []
        # Prepend gap prices to the returned future list so UI will plot from last_date+1.. onwards
        # Normalize gap_prices to plain list
        if hasattr(gap_prices, "tolist"):
            gap_prices_list = gap_prices.tolist()
        elif isinstance(gap_prices, (list, tuple, np.ndarray)):
            gap_prices_list = list(gap_prices)
        else:
            gap_prices_list = [gap_prices]
        # Combine gap fill with immediate horizon forecast
        future_prices_list = gap_prices_list + future_prices_list

    # Map filtered indices back to original df indices
    kept_indices = np.where(kept_mask.values)[0].tolist()
    # Collect the filtered dates aligned to the modeling series (timezone-naive)
    dates_filtered = pd.to_datetime(df["date"]).dt.tz_localize(
        None).iloc[kept_indices].tolist()

    # Derive train/test boundary in original filtered index
    test_start_idx = cfg.window + split_idx  # in filtered index space

    return {
        "kept_indices": kept_indices,
        "train_test_boundary": int(test_start_idx),
        "fitted_on_filtered": fitted_full,  # length == len(close_filtered)
        "filtered_len": int(len(close_filtered)),
        "scaler": type(scaler).__name__ if scaler is not None else None,
        "features": "close",
        "changed_train_indices": changed_idx,
        "future": future_prices_list,
        "horizon_used": int(horizon_used),
        "metrics": m,
        "history": history,
        "model": mk,
        # Expose windowed arrays for reporting (ND array view in UI)
        "X_train": X_train,  # 2D: [n_samples, window]
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "window": int(cfg.window),
        # Unscaled series and dates used for training after filtering/softening
        "close_filtered": close_filtered.tolist(),
        "dates_filtered": dates_filtered,
        "fitted_scaler": fitted_scaler if 'fitted_scaler' in locals() else None,
        "feature_cols": fitted_feature_cols if 'fitted_feature_cols' in locals() else None,
    }


def train_and_forecast_close_only_sklearn(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 1,
    batch_size: int = 32,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train/evaluate using the fast scikit-learn model (no TensorFlow required)."""
    # Thin wrapper that selects the sklearn path in the core training function.
    return _train_and_forecast_core(df, cfg, model_kind="sklearn", epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate, progress_callback=None, callbacks=callbacks)


def train_and_forecast_close_only_lstm(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train/evaluate using an LSTM (requires TensorFlow/Keras)."""
    # Wrapper that invokes the core pipeline with the LSTM model builder.
    return _train_and_forecast_core(df, cfg, model_kind="lstm", epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate, progress_callback=progress_callback, callbacks=callbacks)


def train_and_forecast_close_only_gru(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train/evaluate using a GRU (requires TensorFlow/Keras)."""
    # Wrapper that invokes the core pipeline with the GRU model builder.
    return _train_and_forecast_core(df, cfg, model_kind="gru", epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate, progress_callback=progress_callback, callbacks=callbacks)


def train_and_forecast_close_only_bilstm(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train/evaluate using a bidirectional LSTM (requires TensorFlow/Keras)."""
    # Wrapper selecting the bidirectional LSTM variant in the core trainer.
    return _train_and_forecast_core(df, cfg, model_kind="bilstm", epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate, progress_callback=progress_callback, callbacks=callbacks)


def train_and_forecast_close_only_deep_lstm(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train/evaluate using a deeper (stacked) LSTM (requires TensorFlow/Keras)."""
    # Wrapper selecting the deeper stacked LSTM architecture in the core trainer.
    return _train_and_forecast_core(df, cfg, model_kind="deep-lstm", epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate, progress_callback=progress_callback, callbacks=callbacks)


def train_and_forecast_close_only_deep_transition_lstm(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train/evaluate using a deep-transition LSTM (stacked RNN cells inside one RNN)."""
    return _train_and_forecast_core(df, cfg, model_kind="deep-transition-lstm", epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate, progress_callback=progress_callback, callbacks=callbacks)


def train_and_forecast_close_only_deep_gru(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train/evaluate using a deeper (stacked) GRU (requires TensorFlow/Keras)."""
    # Wrapper selecting the deeper stacked GRU architecture in the core trainer.
    return _train_and_forecast_core(df, cfg, model_kind="deep-gru", epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate, progress_callback=progress_callback, callbacks=callbacks)


def train_and_forecast_multivariate(
    df: pd.DataFrame,
    cfg: ModelConfig,
    model_kind: str = "lstm",
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Low-risk wrapper to force the multifeature (technical) path and delegate to core.

    This mirrors the style of other `train_and_forecast_close_only_*` wrappers but
    sets `cfg.add_technical_features = True` on a shallow copy of the provided
    `ModelConfig` so callers can opt into multivariate training without touching
    the core pipeline.
    """
    import copy

    cfg2 = copy.copy(cfg)
    cfg2.add_technical_features = True
    return _train_and_forecast_core(
        df,
        cfg2,
        model_kind=model_kind,
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        learning_rate=learning_rate,
        progress_callback=progress_callback,
        callbacks=callbacks,
    )


def train_and_forecast_close_only(
    df: pd.DataFrame,
    cfg: ModelConfig,
    model_kind: str = "sklearn",
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[
        int, int, Optional[float]], None]] = None,
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train and forecast using the selected model kind.

    Backward-compatible wrapper. Prefer the specific functions above for readability.
    """
    # Dispatcher that maps a simple model_kind string to the appropriate wrapper.
    # Keeps the public API compact and backward-compatible.
    mapper = {
        "sklearn": train_and_forecast_close_only_sklearn,
        "lstm": train_and_forecast_close_only_lstm,
        "gru": train_and_forecast_close_only_gru,
        "bilstm": train_and_forecast_close_only_bilstm,
        "deep-lstm": train_and_forecast_close_only_deep_lstm,
        "stacked-lstm": train_and_forecast_close_only_deep_lstm,
        "deep-transition-lstm": train_and_forecast_close_only_deep_transition_lstm,
        "deep-gru": train_and_forecast_close_only_deep_gru,
    }
    mk = (model_kind or "sklearn").lower()
    func = mapper.get(mk, train_and_forecast_close_only_sklearn)
    return func(df, cfg, epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate, progress_callback=progress_callback, callbacks=callbacks)


def get_company_name(ticker: str) -> Optional[str]:
    """Best-effort attempt to get the company short name for a ticker."""
    # Try fast_info first, then fall back to Yahoo's quoteSummary API.
    # Returns the short company name or None on failure.
    import json
    try:
        info = yf.Ticker(ticker).fast_info
        name = info.get("shortName") if isinstance(info, dict) else None
        if not name:
            # fallback to quoteSummary API
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=price"
            r = requests.get(url, timeout=10)
            j = r.json()
            name = (
                j.get("quoteSummary", {})
                .get("result", [{}])[0]
                .get("price", {})
                .get("shortName")
            )
        return name
    except requests.RequestException:
        return None
    except json.JSONDecodeError:
        return None
