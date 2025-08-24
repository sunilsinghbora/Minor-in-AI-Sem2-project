import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import importlib
from utils import (
    ModelConfig,
    fetch_prices,
    train_and_forecast_close_only,
    get_company_name,
    search_tickers,
    validate_ticker_symbol,
)

# Detect TensorFlow/Keras availability from utils (optional)
try:
    import utils as U  # type: ignore
    TF_AVAILABLE = getattr(U, "keras", None) is not None
except Exception:
    TF_AVAILABLE = False

# Optional single-box autocomplete component
try:
    from streamlit_searchbox import st_searchbox  # type: ignore
    SEARCHBOX_AVAILABLE = True
except Exception:
    SEARCHBOX_AVAILABLE = False

st.set_page_config(page_title="Stock Forecast (Streamlit)", layout="wide")

def _is_listlike(v):
    return isinstance(v, (list, tuple, np.ndarray))

# Ensure Plotly x inputs are always 1D lists of scalars, aligned with y length
def _sanitize_x(x_vals, y_vals):
    try:
        n = len(y_vals) if y_vals is not None else 0

        # Start with a basic python list
        if isinstance(x_vals, np.ndarray):
            x_vals = x_vals.tolist()

        # If x is a pandas Series/Index, convert to list
        try:
            import pandas as _pd  # local import to avoid global dependency at import time
            if isinstance(x_vals, (_pd.Series, _pd.Index)):
                x_vals = x_vals.tolist()
        except Exception:
            pass

        # If x is a single nested container like [[...]] or ([(...)]) -> flatten to inner
        if _is_listlike(x_vals) and len(x_vals) == 1 and _is_listlike(x_vals[0]):
            try:
                x_vals = list(np.ravel(x_vals[0]).tolist())
            except Exception:
                x_vals = list(x_vals[0])

        # If x is list-like and any element is itself list-like, give up and use positional index
        if _is_listlike(x_vals):
            for elem in x_vals[: min(5, len(x_vals))]:  # sample few
                if _is_listlike(elem):
                    return list(range(n))
        else:
            # Not list-like: build positional index
            return list(range(n))

        # Length guard
        if len(x_vals) != n:
            return list(range(n))

        # Ensure scalars are JSON-serializable (convert numpy types)
        out = []
        for v in x_vals:
            if isinstance(v, (np.generic,)):
                v = v.item()
            out.append(v)
        return out
    except Exception:
        return list(range(len(y_vals) if y_vals is not None else 0))

def _fix_figure_x(fig: go.Figure) -> go.Figure:
    try:
        for tr in fig.data:
            x = getattr(tr, "x", None)
            y = getattr(tr, "y", None)
            if y is None:
                continue
            new_x = _sanitize_x(x, y)
            # If new_x is numpy array, convert to list for safety
            if isinstance(new_x, np.ndarray):
                new_x = new_x.tolist()
            tr.update(x=new_x)
    except Exception:
        pass
    return fig

def _safe_plot(fig: go.Figure):
    try:
        st.plotly_chart(_fix_figure_x(fig), use_container_width=True)
    except Exception:
        # Fallback: rebuild traces with x as positional indices
        try:
            fig_fallback = go.Figure()
            for tr in fig.data:
                y = getattr(tr, "y", None)
                if y is None:
                    continue
                y1 = _flatten_1d(y)
                xf = list(range(len(y1)))
                fig_fallback.add_trace(go.Scatter(x=xf, y=y1, mode=getattr(tr, "mode", "lines"), name=getattr(tr, "name", None), line=getattr(tr, "line", None)))
            fig_fallback.update_layout(template="plotly_white", height=fig.layout.height or 400, legend_orientation=fig.layout.legend.orientation if fig.layout.legend else "h", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_fallback, use_container_width=True)
        except Exception as e:
            st.error(str(e))

def _flatten_1d(arr):
    try:
        if isinstance(arr, np.ndarray):
            arr = np.squeeze(arr)
            if arr.ndim > 1:
                arr = arr.ravel()
            return arr.tolist()
        if isinstance(arr, tuple):
            arr = list(arr)
        if isinstance(arr, list):
            if len(arr) > 0 and isinstance(arr[0], (list, tuple, np.ndarray)):
                try:
                    return np.array(arr).ravel().tolist()
                except Exception:
                    flat = []
                    for it in arr:
                        if isinstance(it, (list, tuple)):
                            flat.extend(list(it))
                        elif isinstance(it, np.ndarray):
                            flat.extend(np.ravel(it).tolist())
                        else:
                            flat.append(it)
                    return flat
            return arr
        return arr
    except Exception:
        return arr

def _add_trace_safe(fig: go.Figure, x, y, **kwargs):
    # Flatten and sanitize y first
    y1d = _flatten_1d(y)
    # Build a robust x vector
    x_in = _flatten_1d(x)
    x1d = _sanitize_x(x_in, y1d)
    try:
        fig.add_trace(go.Scatter(x=x1d, y=y1d, **kwargs))
    except Exception:
        # Absolute fallback to positional x
        xf = list(range(len(y1d)))
        try:
            fig.add_trace(go.Scatter(x=xf, y=y1d, **kwargs))
        except Exception:
            # If even that fails (shouldn't), drop styling to bare minimum
            fig.add_trace(go.Scatter(x=xf, y=y1d))

# Session state init
if "analysis" not in st.session_state:
    st.session_state["analysis"] = None
if "pending_run" not in st.session_state:
    st.session_state["pending_run"] = None

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    # Single-box autocomplete if available, else fallback to text + matches
    ticker = "MSFT"
    if SEARCHBOX_AVAILABLE:
        def _search_fn(term: str):
            res = search_tickers(term or "", limit=8)
            return [f"{s['symbol']} — {s.get('name','')} ({s.get('exchDisp','')})" for s in res]

        picked_label = st_searchbox(
            _search_fn,
            key="ticker_searchbox",
            default="MSFT — Microsoft Corporation (NasdaqGS)",
            placeholder="Type symbol or company name",
        )
        if picked_label:
            ticker = picked_label.split(" — ")[0].strip().upper()
    else:
        # Fallback: Search-as-you-type with separate Matches list
        ti = st.text_input("Search ticker or company", value="MSFT", help="Type symbol (e.g., MSFT) or company name")
        query = (ti or "").strip()
        suggestions = search_tickers(query, limit=8) if len(query) >= 2 else []
        formats = [f"{s['symbol']} — {s.get('name','')} ({s.get('exchDisp','')})" for s in suggestions]
        picked = st.selectbox("Matches", options=["Use typed value"] + formats, index=0) if formats else None
        if picked and picked != "Use typed value":
            sel = suggestions[formats.index(picked)]
            ticker = sel["symbol"].upper()
        else:
            ticker = query.upper()

    # Validate ticker before allowing run
    is_valid = validate_ticker_symbol(ticker) if len(ticker) >= 1 else False
    if ticker and not is_valid:
        st.warning("Ticker looks invalid or has no recent data. Please pick a valid symbol.")
    period = st.selectbox(
        "History (training)",
        ["1y", "2y", "5y", "10y", "20y", "25y", "max"],
        index=0,
        help="Training uses at least 1 year. Use Quick view below to zoom chart.",
    )

    st.subheader("Model")
    model_options = ["sklearn (fast)"] if not TF_AVAILABLE else [
        "sklearn (fast)",
        "lstm",
        "gru",
        "bilstm",
        "bigru",
        "deep-lstm",
        "deep-gru",
    ]
    model_kind = st.selectbox(
        "Model",
        model_options,
        index=0,
        help=("Neural models require TensorFlow/Keras" + (" (currently unavailable)" if not TF_AVAILABLE else "")),
    )
    if not TF_AVAILABLE:
        st.caption("Neural models are disabled because TensorFlow is not available. Install a compatible TF to enable them.")
    window = st.slider("Window", 10, 120, 30, 1)
    horizon = st.slider("Horizon (days)", 1, 120, 10, 1)
    test_split = st.slider("Test split (%)", 5, 30, 10, 1) / 100.0
    epochs = st.slider("Epochs (neural models)", 5, 100, 30, 5)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    dropout = st.slider("Dropout", 0.0, 0.8, 0.3, 0.05)

    st.subheader("Outliers")
    filter_outliers = st.checkbox("Enable outlier filter (±%)", value=False)
    outlier_threshold = st.slider("Outlier threshold % (inclusive)", 3, 20, 5, 1)

    st.subheader("Spike softening (train only)")
    soften = st.checkbox("Enable softening", value=False)
    spike_threshold = st.slider("Spike threshold %", 5, 25, 10, 1)
    spike_factor = st.slider("Spike factor", 0.1, 0.9, 0.5, 0.1)

    # Display options removed per request: no quick view or debug toggles

    run_btn = st.button("Run analysis", type="primary", disabled=not is_valid)

st.title("Stock Forecast")

# Fresh run trigger: capture inputs, clear state, rerun
if run_btn and ticker:
    params = {
        "ticker": ticker,
        "period": period,
        "model_kind": model_kind,
        "window": window,
        "horizon": horizon,
        "test_split": test_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout": dropout,
        "filter_outliers": filter_outliers,
        "outlier_threshold": float(outlier_threshold),
        "soften_spikes": soften,
        "spike_threshold": float(spike_threshold),
        "spike_factor": float(spike_factor),
    }
    st.session_state.clear()
    st.session_state["pending_run"] = params
    st.rerun()

# Execute pending run on a clean state
if st.session_state.get("pending_run"):
    p = st.session_state.pop("pending_run")
    try:
        with st.spinner("Fetching data…"):
            df = fetch_prices(p["ticker"], period=p["period"])
        name = get_company_name(p["ticker"]) or ""
        st.caption(f"{p['ticker']} • {name}")

        cfg = ModelConfig(
            window=p["window"],
            horizon=p["horizon"],
            test_split=p["test_split"],
            filter_outliers=p["filter_outliers"],
            outlier_threshold=float(p["outlier_threshold"]),
            soften_spikes=p["soften_spikes"],
            spike_threshold=float(p["spike_threshold"]),
            spike_factor=float(p["spike_factor"]),
        )

        mk = p["model_kind"].split()[0] if p["model_kind"].startswith("sklearn") else p["model_kind"]
        progress = st.progress(0, text="Training…")
        status = st.empty()

        def on_progress(done: int, total: int, loss: float | None):
            pct = int(100 * done / max(1, total))
            msg = f"Epoch {done}/{total}"
            if loss is not None:
                msg += f" • loss: {loss:.4f}"
            progress.progress(pct, text=msg)
            status.write(msg)

        result = train_and_forecast_close_only(
            df,
            cfg,
            model_kind=mk,
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            dropout=p["dropout"],
            progress_callback=on_progress if mk != "sklearn" else None,
        )
        progress.empty()
        status.empty()

        # Save fresh analysis
        st.session_state["analysis"] = {
            "ticker": p["ticker"],
            "period": p["period"],
            "name": name,
            "df": df,
            "result": result,
            "params": {
                "test_split": p["test_split"],
                "epochs": p["epochs"],
                "batch_size": p["batch_size"],
                "dropout": p["dropout"],
                "filter_outliers": p["filter_outliers"],
                "outlier_threshold": p["outlier_threshold"],
                "soften_spikes": p["soften_spikes"],
                "spike_threshold": p["spike_threshold"],
                "spike_factor": p["spike_factor"],
            },
        }
    except Exception as e:
        st.error(str(e))

# Render from stored analysis so quick-range buttons work without retraining
analysis = st.session_state.get("analysis")
if analysis:
    df = analysis["df"]
    result = analysis["result"]
    ticker = analysis["ticker"]
    name = analysis["name"]
    st.caption(f"{ticker} • {name}")

    kept = result["kept_indices"]
    fitted = result["fitted_on_filtered"]
    boundary = result["train_test_boundary"]
    future = result["future"]

    # Build training subset used for fitting
    try:
        train_idx_original = kept[: boundary] if boundary <= len(kept) else list(kept)
        training_df = df.iloc[train_idx_original].reset_index(drop=True)
    except Exception:
        training_df = df.copy().reset_index(drop=True)

    # Dates for plotting
    dates_series = pd.to_datetime(df["date"]).dt.tz_localize(None)
    dates = dates_series.dt.strftime("%Y-%m-%d").tolist()
    prices = df["price"].values.astype(float)

    fitted_on_original = [np.nan] * len(df)
    for j, idx in enumerate(kept):
        if j < len(fitted):
            fitted_on_original[idx] = fitted[j]
    test_start_original = kept[boundary] if boundary < len(kept) else kept[-1]

    if len(dates_series) >= 2:
        step = dates_series.iloc[1] - dates_series.iloc[0]
    else:
        step = pd.Timedelta(days=1)
    last_date = dates_series.iloc[-1]
    future_dt = [last_date + (i + 1) * step for i in range(len(future))]
    future_dates = [d.strftime("%Y-%m-%d") for d in future_dt]

    # Main chart
    fig = go.Figure()
    # Debug captions removed per request
    _add_trace_safe(fig, dates, prices, mode="lines", name="Actual", line=dict(color="#000000", width=2))
    train_mask = [i <= test_start_original for i in range(len(df))]
    test_mask = [i > test_start_original for i in range(len(df))]
    _add_trace_safe(fig, dates, [fitted_on_original[i] if train_mask[i] else None for i in range(len(df))], mode="lines", name="Training fit", line=dict(color="#FFA500", width=2))
    _add_trace_safe(fig, dates, [fitted_on_original[i] if test_mask[i] else None for i in range(len(df))], mode="lines", name="Testing fit", line=dict(color="#2E8B57", width=2))
    _add_trace_safe(fig, future_dates, future, mode="lines", name="Future", line=dict(color="#FF0000", width=2, dash="dot"))
    fig.update_layout(template="plotly_white", height=550, legend_orientation="h", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
    _safe_plot(fig)

    # Quick view range removed per request

    # Panels
    st.subheader("Used config")
    p = analysis["params"]
    st.json({
        "features": "close",
        "scaler": "StandardScaler",
        "test_split": p.get("test_split"),
        "model": result.get("model"),
        "epochs": p.get("epochs"),
        "batch_size": p.get("batch_size"),
        "dropout": p.get("dropout"),
        "filter_outliers": p.get("filter_outliers"),
        "outlier_threshold": p.get("outlier_threshold"),
        "soften_spikes": p.get("soften_spikes"),
        "spike_threshold": p.get("spike_threshold"),
        "spike_factor": p.get("spike_factor"),
    })

    with st.expander("Model metrics", expanded=True):
        m = result.get("metrics") or {}
        cols = st.columns(4)
        cols[0].metric("RMSE", f"{m.get('RMSE', float('nan')):.2f}")
        cols[1].metric("MAE", f"{m.get('MAE', float('nan')):.2f}")
        cols[2].metric("MAPE (%)", f"{m.get('MAPE', float('nan')):.2f}")
        cols[3].metric("R²", f"{m.get('R2', float('nan')):.3f}")
        hist = result.get("history") or {}
        if hist and (hist.get("loss") or hist.get("val_loss")):
            ep = list(range(1, max(len(hist.get("loss", [])), len(hist.get("val_loss", []))) + 1))
            figh = go.Figure()
            if hist.get("loss"):
                y_loss = list(hist["loss"])
                _add_trace_safe(figh, ep, y_loss, mode="lines", name="loss", line=dict(color="#1f77b4"))
            if hist.get("val_loss"):
                y_vloss = list(hist["val_loss"])
                _add_trace_safe(figh, ep, y_vloss, mode="lines", name="val_loss", line=dict(color="#ff7f0e"))
            figh.update_layout(height=250, template="plotly_white", title="Training history")
            _safe_plot(figh)
        # Offer CSV download for the exact training subset used
        try:
            csv_bytes = training_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download training CSV",
                data=csv_bytes,
                file_name=f"{ticker}_{analysis['period']}_training.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception:
            pass
