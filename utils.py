from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow is optional at import time. We only require it if a neural model is selected.
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - allow running without TF installed
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore

"""Sentiment-related utilities removed by request."""


@dataclass
class ModelConfig:
    window: int = 30
    horizon: int = 10
    test_split: float = 0.1
    filter_outliers: bool = False
    outlier_threshold: float = 5.0  # percent, inclusive
    soften_spikes: bool = False
    spike_threshold: float = 10.0  # percent, inclusive
    spike_factor: float = 0.5      # new = prev + factor * delta


def fetch_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    code = (period or "1y").lower()
    # Map extended codes; yfinance doesn't support 20y/25y directly, so filter from max
    if code in {"1w", "1week"}:
        start = (dt.datetime.utcnow() - dt.timedelta(days=7)).date()
        df = t.history(start=start, interval="1d", auto_adjust=False)
    elif code in {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd"}:
        df = t.history(period=code, interval="1d", auto_adjust=False)
    elif code in {"20y", "25y"}:
        df = t.history(period="max", interval="1d", auto_adjust=False)
        years = 20 if code == "20y" else 25
        cutoff = (dt.datetime.utcnow() - dt.timedelta(days=365 * years)).date()
        # Will filter after reset_index/rename
    elif code in {"max"}:
        df = t.history(period="max", interval="1d", auto_adjust=False)
    else:
        # Fallback
        df = t.history(period="1y", interval="1d", auto_adjust=False)
    if df.empty:
        raise RuntimeError("No data from Yahoo Finance")
    df = df.reset_index()
    df.rename(columns={"Date": "date", "Close": "price", "Open": "open"}, inplace=True)
    df = df[["date", "open", "price"]].dropna()
    # Ensure timezone-naive datetimes for safe comparisons
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    # Apply 20y/25y cutoff if needed
    if code in {"20y", "25y"}:
        years = 20 if code == "20y" else 25
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * years)
        df = df[df["date"] >= cutoff]
    return df


def search_tickers(query: str, limit: int = 10, region: str = "US", lang: str = "en-US") -> List[Dict[str, Any]]:
    """
    Lightweight symbol search using Yahoo Finance public search endpoint.
    Returns a list of dicts: {symbol, name, exch, type, exchDisp} filtered to likely equities/etfs.
    """
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
            name = it.get("shortname") or it.get("longname") or it.get("quoteType") or ""
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
    """Quick validation by fetching minimal history; returns True if data exists."""
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
    # True means keep
    pct = series.pct_change() * 100.0
    mask = pct.abs().le(pct_threshold) | pct.isna()  # keep first NaN
    # Always keep the first value
    mask.iloc[0] = True
    return mask


def soften_spikes_train_only(series: pd.Series, train_end_idx: int, threshold: float, factor: float) -> Tuple[pd.Series, List[int]]:
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


def make_windows_close_only(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window : i])
        y.append(values[i])
    return np.array(X), np.array(y)

def _build_keras_model(input_len: int, kind: str = "lstm", dropout: float = 0.3) -> Any:
    if keras is None or layers is None:
        raise RuntimeError(
            "Neural models require TensorFlow/Keras. Select 'sklearn (fast)' or install a compatible TensorFlow package."
        )
    kind = (kind or "lstm").lower()
    is_gru = "gru" in kind
    bi = kind.startswith("bi") or kind in {"bilstm", "bigru", "bidirectional lstm", "bidirectional gru"}
    deep = kind.startswith("deep") or ("deep" in kind)

    seq = keras.Sequential()
    seq.add(layers.Input(shape=(input_len, 1)))
    RNN = layers.GRU if is_gru else layers.LSTM
    if bi:
        seq.add(layers.Bidirectional(RNN(64, return_sequences=deep)))
    else:
        seq.add(RNN(64, return_sequences=deep))
    seq.add(layers.Dropout(dropout))
    if deep:
        if bi:
            seq.add(layers.Bidirectional(RNN(32)))
        else:
            seq.add(RNN(32))
        seq.add(layers.Dropout(dropout))
    seq.add(layers.Dense(1))
    seq.compile(optimizer="adam", loss="mse")
    return seq

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
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
    close_test = close_filtered.iloc[train_n:]

    changed_idx: List[int] = []
    if cfg.soften_spikes:
        soft_train, changed = soften_spikes_train_only(
            close_filtered, train_end_idx=train_n - 1, threshold=cfg.spike_threshold, factor=cfg.spike_factor
        )
        close_train = soft_train.iloc[:train_n]
        changed_idx = [i for i in changed if i < train_n]

    # Scale
    scaler = StandardScaler()
    train_vals = close_train.values.reshape(-1, 1)
    scaler.fit(train_vals)
    full_scaled = scaler.transform(close_filtered.values.reshape(-1, 1)).flatten()

    # Windows on scaled series
    X_all, y_all = make_windows_close_only(full_scaled, cfg.window)
    # determine split index in windowed space
    split_idx = max(1, train_n - cfg.window)
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test, y_test = X_all[split_idx:], y_all[split_idx:]

    history = None
    mk = (model_kind or "sklearn").lower()
    if mk != "sklearn":
        if keras is None:
            # Fail fast with a clear message if user selects a neural model without TF available
            raise RuntimeError(
                "TensorFlow/Keras is not available in this environment. Install TensorFlow (CPU) compatible with Python 3.12 (e.g., 2.16.1) or switch to 'sklearn (fast)'."
            )
        # reshape to [samples, timesteps, features]
        Xtr = X_train[..., None]
        Xte = X_test[..., None]
        Xall = X_all[..., None]
        model = _build_keras_model(cfg.window, mk, dropout=dropout)
        # Train one epoch at a time to report progress
        total_epochs = int(max(1, epochs))
        history = {"loss": [], "val_loss": []}
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
            vl = float(h.history.get("val_loss", [np.nan])[-1]) if len(Xte) else np.nan
            history["loss"].append(l)
            if not np.isnan(vl):
                history["val_loss"].append(vl)
            if callable(progress_callback):
                try:
                    progress_callback(ep + 1, total_epochs, l)
                except Exception:
                    pass
        y_pred_all = model.predict(Xall, verbose=0).flatten()
    else:
        # Fast, CPU-friendly regressor
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        # Fitted (train+test) backcast
        y_pred_all = model.predict(X_all)

    # Metrics on test
    def inv(v):
        return scaler.inverse_transform(np.array(v).reshape(-1, 1)).flatten()

    fitted_prices = inv(y_pred_all)
    actual_prices = inv(y_all)
    # Metrics on test region (inverse scaled)
    if len(y_test):
        y_test_inv = inv(y_test)
        y_pred_test_inv = inv(y_pred_all[len(y_train) : len(y_train) + len(y_test)])
        m = _metrics(y_test_inv, y_pred_test_inv)
    else:
        m = {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan"), "R2": float("nan")}

    # build aligned series to original filtered timeline
    fitted_full = [np.nan] * len(full_scaled)
    # windows produce predictions starting at index cfg.window
    for i, val in enumerate(fitted_prices, start=cfg.window):
        fitted_full[i] = val

    # Future forecasting (iterative)
    last_window = full_scaled[-cfg.window :].tolist()
    future_scaled = []
    for _ in range(cfg.horizon):
        if mk != "sklearn":
            # Keras models expect [batch, timesteps, features]
            x_in = np.array(last_window, dtype=float).reshape(1, cfg.window, 1)
            pred = model.predict(x_in, verbose=0)
            next_scaled = float(np.ravel(pred)[0])
        else:
            # Sklearn models trained on 2D [samples, timesteps]
            next_scaled = float(model.predict([last_window])[0])
        future_scaled.append(next_scaled)
        last_window = last_window[1:] + [next_scaled]
    future_prices = inv(future_scaled)

    # Map filtered indices back to original df indices
    kept_indices = np.where(kept_mask.values)[0].tolist()

    # Derive train/test boundary in original filtered index
    test_start_idx = cfg.window + split_idx  # in filtered index space

    return {
        "kept_indices": kept_indices,
        "train_test_boundary": int(test_start_idx),
        "fitted_on_filtered": fitted_full,  # length == len(close_filtered)
        "filtered_len": int(len(close_filtered)),
        "scaler": "StandardScaler",
        "features": "close",
        "changed_train_indices": changed_idx,
        "future": future_prices.tolist(),
        "metrics": m,
        "history": history,
        "model": mk,
    }


def train_and_forecast_close_only_sklearn(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 1,
    batch_size: int = 32,
    dropout: float = 0.0,
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    return _train_and_forecast_core(df, cfg, model_kind="sklearn", epochs=epochs, batch_size=batch_size, dropout=dropout, progress_callback=None)


def train_and_forecast_close_only_lstm(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    return _train_and_forecast_core(df, cfg, model_kind="lstm", epochs=epochs, batch_size=batch_size, dropout=dropout, progress_callback=progress_callback)


def train_and_forecast_close_only_gru(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    return _train_and_forecast_core(df, cfg, model_kind="gru", epochs=epochs, batch_size=batch_size, dropout=dropout, progress_callback=progress_callback)


def train_and_forecast_close_only_bilstm(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    return _train_and_forecast_core(df, cfg, model_kind="bilstm", epochs=epochs, batch_size=batch_size, dropout=dropout, progress_callback=progress_callback)


def train_and_forecast_close_only_bigru(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    return _train_and_forecast_core(df, cfg, model_kind="bigru", epochs=epochs, batch_size=batch_size, dropout=dropout, progress_callback=progress_callback)


def train_and_forecast_close_only_deep_lstm(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    return _train_and_forecast_core(df, cfg, model_kind="deep-lstm", epochs=epochs, batch_size=batch_size, dropout=dropout, progress_callback=progress_callback)


def train_and_forecast_close_only_deep_gru(
    df: pd.DataFrame,
    cfg: ModelConfig,
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    return _train_and_forecast_core(df, cfg, model_kind="deep-gru", epochs=epochs, batch_size=batch_size, dropout=dropout, progress_callback=progress_callback)


def train_and_forecast_close_only(
    df: pd.DataFrame,
    cfg: ModelConfig,
    model_kind: str = "sklearn",
    epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    progress_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper. Prefer calling the specific functions above for readability."""
    mapper = {
        "sklearn": train_and_forecast_close_only_sklearn,
        "lstm": train_and_forecast_close_only_lstm,
        "gru": train_and_forecast_close_only_gru,
        "bilstm": train_and_forecast_close_only_bilstm,
        "bigru": train_and_forecast_close_only_bigru,
        "deep-lstm": train_and_forecast_close_only_deep_lstm,
        "deep-gru": train_and_forecast_close_only_deep_gru,
    }
    mk = (model_kind or "sklearn").lower()
    func = mapper.get(mk, train_and_forecast_close_only_sklearn)
    return func(df, cfg, epochs=epochs, batch_size=batch_size, dropout=dropout, progress_callback=progress_callback)


def get_company_name(ticker: str) -> Optional[str]:
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
