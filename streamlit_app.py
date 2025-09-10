import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

# Small helper: test whether a value is a list-like container.
# Accepts Python lists/tuples and NumPy arrays so callers can branch quickly.
# Used to detect nested sequence inputs in plotting helpers.


def _is_listlike(v):
    return isinstance(v, (list, tuple, np.ndarray))

# Ensure Plotly x inputs are always 1D lists of scalars, aligned with y length
# Normalize a candidate x-axis input for Plotly traces.
# Converts Series/Index/ndarray to plain lists, flattens trivial nesting,
# and falls back to positional indices when lengths or shapes don't align.


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
    # Walk every trace in a Plotly figure and ensure the trace.x array
    # is a 1-D list aligned with its y values. Converts numpy arrays to
    # Python lists for safe JSON serialization before updating traces.
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
    # Safely render a Plotly figure in Streamlit by first normalizing
    # x arrays; if that fails, rebuild traces with positional x indices
    # so a fallback chart still appears instead of crashing the app.
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
                fig_fallback.add_trace(go.Scatter(x=xf, y=y1, mode=getattr(
                    tr, "mode", "lines"), name=getattr(tr, "name", None), line=getattr(tr, "line", None)))
            fig_fallback.update_layout(template="plotly_white", height=fig.layout.height or 400,
                                       legend_orientation=fig.layout.legend.orientation if fig.layout.legend else "h", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_fallback, use_container_width=True)
        except Exception as e:
            st.error(str(e))


def _flatten_1d(arr):
    # Convert various array-like inputs into a flat Python list.
    # Handles numpy arrays, tuples, nested lists, and ragged sequences.
    # Returns the input unchanged when it cannot be reasonably flattened.
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
    # Add a Scatter trace robustly: flatten inputs and fall back to
    # positional x indices if Plotly rejects the provided x values.
    # This keeps plotting resilient to ragged or mixed-type sequences.
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
    last_ticker = st.session_state.get("ticker", "")
    ticker = last_ticker
    if SEARCHBOX_AVAILABLE:
        def _search_fn(term: str):
            res = search_tickers(term or "", limit=8)
            return [f"{s['symbol']} — {s.get('name','')} ({s.get('exchDisp','')})" for s in res]
        # Use previously chosen label if any; otherwise leave empty
        default_label = st.session_state.get("ticker_label", "")
        picked_label = st_searchbox(
            _search_fn,
            key="ticker_searchbox",
            default=default_label,
            placeholder="Type symbol or company name",
        )
        if picked_label:
            ticker = picked_label.split(" — ")[0].strip().upper()
            st.session_state["ticker_label"] = picked_label
            st.session_state["ticker"] = ticker
    else:
        # Fallback: Search-as-you-type with separate Matches list
        ti = st.text_input("Search ticker or company", value=last_ticker,
                           help="Type symbol (e.g., MSFT) or company name")
        query = (ti or "").strip()
        suggestions = search_tickers(query, limit=8) if len(query) >= 2 else []
        formats = [
            f"{s['symbol']} — {s.get('name','')} ({s.get('exchDisp','')})" for s in suggestions]
        picked = st.selectbox(
            "Matches",
            options=["Use typed value"] + formats,
            index=0,
            help="Choose a suggested ticker/company from the search results, or pick 'Use typed value' to use your typed query.",
        ) if formats else None
        if picked and picked != "Use typed value":
            sel = suggestions[formats.index(picked)]
            ticker = sel["symbol"].upper()
        else:
            ticker = query.upper()
        st.session_state["ticker"] = ticker

    # Validate ticker before allowing run
    is_valid = validate_ticker_symbol(ticker) if len(ticker) >= 1 else False
    if ticker and not is_valid:
        st.warning(
            "Ticker looks invalid or has no recent data. Please pick a valid symbol.")

    # Preload full history for date bounds
    df_all = None
    min_date = None
    max_date = None
    default_start = None
    default_end = None
    if is_valid:
        try:
            df_all = fetch_prices(ticker, period="max")
            dates_series_sb = pd.to_datetime(
                df_all["date"]).dt.tz_localize(None)
            min_date = dates_series_sb.iloc[0].date()
            max_date = dates_series_sb.iloc[-1].date()
            # Default to last 10 years (clamped to available min)
            ten_years_ago = (
                dates_series_sb.iloc[-1] - pd.DateOffset(years=10)).date()
            default_start = ten_years_ago if ten_years_ago > min_date else min_date
            default_end = max_date
        except Exception:
            pass

    st.subheader("Model")
    # Neural model options: include both shallow LSTM and stacked/deep variants
    neural_options = ["lstm", "stacked-lstm", "deep-lstm", "gru", "bilstm"]
    if TF_AVAILABLE:
        # Move the model hints into the selectbox `help` so Streamlit shows the
        # small built-in '?' tooltip next to the control.
        model_hints = (
            "Window controls how many past days the model sees; larger windows capture longer patterns but need more data. "
            "Horizon is forecast length and is capped at 14 to reduce compounding error. "
            "Epochs/batch/dropout affect neural training (more epochs = longer but may overfit; higher dropout = more regularization)."
        )
        model_kind = st.selectbox(
            "Model (neural)",
            neural_options,
            index=0,
            help=("Neural models require TensorFlow/Keras" +
                  (" (currently unavailable)" if not TF_AVAILABLE else "") +
                  "\n\n" + model_hints),
        )
    else:
        st.caption(
            "Neural models are disabled because TensorFlow is not available. Install a compatible TF to enable them.")
        model_kind = None

    # sklearn (fast) is used implicitly as a fallback when TensorFlow is not available

    # model hint moved into selectbox `help` above; no separate info button required.

    with st.expander("Model & training settings", expanded=True):
        window_options = [3, 5] + list(range(10, 95, 5))  # 10..90
        default_window = 30 if 30 in window_options else window_options[0]
        window = st.selectbox(
            "Window",
            window_options,
            index=window_options.index(default_window),
            help="How many past days the model sees as input (larger windows need more data).",
        )
        # Forecast horizon (1..14 days)
        # default horizon set to 7 days
        horizon = st.selectbox("Horizon (days)", options=list(
            range(1, 15)), index=6, help="Forecast length in days (1..14).")
        test_split = st.slider("Test split (%)", 5, 30, 10, 1,
                               help="Percent of data reserved for testing (5-30%).") / 100.0
        epochs = st.slider("Epochs (neural models)", 5, 100, 30, 5,
                           help="Number of training epochs for neural models (more epochs = longer training).")
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1,
                                  help="Mini-batch size used during training; larger batches use more memory.")
        dropout = st.slider("Dropout", 0.0, 0.8, 0.3, 0.05,
                            help="Dropout rate applied during training to reduce overfitting (0=no dropout).")
        # --- Training dates moved into this expander per request ---
        st.subheader("Training dates")

        def _clamp_date(val, minv, maxv):
            try:
                if val is None:
                    return None
                v = pd.to_datetime(val).date()
                if minv and v < minv:
                    return minv
                if maxv and v > maxv:
                    return maxv
                return v
            except Exception:
                return None

        if min_date and max_date:
            # Default to last 10 years when possible, otherwise use earliest available
            ten_years_ago = (pd.to_datetime(df_all["date"]).dt.tz_localize(
                None).iloc[-1] - pd.DateOffset(years=10)).date()
            start_default = st.session_state.get(
                "start_date", ten_years_ago if ten_years_ago > min_date else min_date)
            end_default = st.session_state.get("end_date", max_date)
            start_default = _clamp_date(
                start_default, min_date, max_date) or min_date
            end_default = _clamp_date(
                end_default, min_date, max_date) or max_date

            start_date = st.date_input(
                "Start date",
                value=start_default,
                min_value=min_date,
                max_value=max_date,
                key="start_date_input",
                help="Select the first date to include in training. The selected range must contain enough rows for the chosen window.",
            )
            end_date = st.date_input(
                "End date",
                value=end_default,
                min_value=min_date,
                max_value=max_date,
                key="end_date_input",
                help="Select the last date to include in training. Make sure Start <= End and the range covers at least window+5 rows.",
            )
        else:
            start_date = None
            end_date = None
    # Advanced (neural) options
    learning_rate = 1e-4  # default 0.1e-3 as requested
    if TF_AVAILABLE and model_kind != "sklearn (fast)":
        with st.expander("Advanced (neural)", expanded=False):
            lr_labels = ["Low (1e-4)", "Default (1e-3)", "High (3e-3)"]
            lr_map = {"Low (1e-4)": 1e-4, "Default (1e-3)": 1e-3,
                      "High (3e-3)": 3e-3}
            sel = st.selectbox("Learning rate", lr_labels, index=0,
                               help="Step size for optimizer updates. Keep Low/Default unless you know you need faster/slower training.")
            learning_rate = float(lr_map.get(sel, 1e-4))
            # Callback options
            enable_early_stopping = st.checkbox("Enable EarlyStopping (neural)", value=True,
                                                help="Stop training early when validation loss stops improving.")
            enable_checkpoint = st.checkbox("Enable ModelCheckpoint (neural)", value=True,
                                            help="Save best model weights during training to a temporary checkpoint.")
            force_retrain = st.checkbox("Force retrain (ignore cache)", value=False,
                                        help="When checked, ignore any in-memory cache and retrain the model.")

    # Training dates are handled inside the 'Model & training settings' expander above.

    dates_ok = True
    range_len = None
    if df_all is not None and start_date and end_date:
        if start_date > end_date:
            st.error("Start date cannot be after end date.")
            dates_ok = False
        # Count rows in the selected range
        try:
            mask_sb = (
                (pd.to_datetime(df_all["date"]).dt.tz_localize(None) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(df_all["date"]).dt.tz_localize(
                    None) <= pd.to_datetime(end_date))
            )
            range_len = int(mask_sb.sum())
            # Require at least window + 5 samples for minimal training
            min_needed = max(30, int(window) + 5)
            if range_len < min_needed:
                st.warning(
                    f"Selected range has only {range_len} rows; needs at least {min_needed} for window={window}.")
                dates_ok = False
        except Exception:
            pass
    else:
        # If we couldn't load dates, keep disabled
        dates_ok = False if is_valid else False

    # Persist in session
    if start_date and end_date:
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date

    with st.expander("Preprocessing & Feature engineering", expanded=False):
        st.subheader("Outliers")
        filter_outliers = st.checkbox(
            "Enable outlier filter (±%)",
            value=False,
            help="Drop days with large day-over-day moves from training to reduce noise; enable if your data may contain errors or you want smoother training.",
        )
        outlier_threshold = st.slider(
            "Outlier threshold % (inclusive)", 3, 20, 5, 1)

        st.subheader("Spike softening (train only)")
        soften = st.checkbox(
            "Enable softening",
            value=False,
            help="Softens large spikes only in the training data (keeps original series for plotting). Use when extreme single-day moves are likely measurement errors.",
        )
        spike_threshold = st.slider("Spike threshold %", 5, 25, 10, 1)
        spike_factor = st.slider("Spike factor", 0.1, 0.9, 0.5, 0.1)

        # Technical features grouped together: rolling mean, single SMA choice, fixed RSI(14)
        st.subheader("Technical features (optional)")
        add_rolling_mean = st.checkbox("Add rolling mean", value=False,
                                       help="Augment inputs with a rolling mean (safe, low-risk).")
        rolling_window = st.number_input(
            "Rolling mean window", min_value=2, max_value=60, value=7, step=1)

        add_sma = st.checkbox(
            "Add SMA", value=False, help="Enable a single SMA feature (choose window below)")
        # No 'Off' option here; SMA window choices default to 20
        sma_options = [10, 20, 50, 200]
        sma_choice = st.selectbox(
            "SMA window", options=sma_options, index=sma_options.index(20))

        add_rsi = st.checkbox("Add RSI (14)", value=False,
                              help="Add 14-day RSI (fixed)")

        # Preview features button: safe, non-invasive preview based on selected date range
        if st.button("Preview features"):
            if df_all is None:
                st.warning(
                    "No data loaded yet for the selected ticker/date range.")
            else:
                # Slice by selected start/end if available
                try:
                    sdt = st.session_state.get("start_date")
                    edt = st.session_state.get("end_date")
                    if sdt and edt:
                        start = pd.to_datetime(sdt)
                        end = pd.to_datetime(edt)
                        df_preview = df_all[(pd.to_datetime(df_all["date"]).dt.tz_localize(None) >= start.tz_localize(None)) & (
                            pd.to_datetime(df_all["date"]).dt.tz_localize(None) <= end.tz_localize(None))].reset_index(drop=True)
                    else:
                        df_preview = df_all.copy()
                except Exception:
                    df_preview = df_all.copy()

                close_proc = df_preview["price"].astype(
                    float).reset_index(drop=True)
                # Include date for preview and sorting (keep original tz-naive strings)
                df_feats = pd.DataFrame({
                    "date": pd.to_datetime(df_preview["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d"),
                    "close": close_proc.values,
                })
                if add_rolling_mean:
                    rw = max(1, int(rolling_window))
                    df_feats["rolling_mean"] = close_proc.rolling(
                        window=rw, min_periods=1).mean().values

                # Include optional SMA and RSI indicators when selected
                if add_sma and int(sma_choice) > 0:
                    df_feats["sma"] = close_proc.rolling(
                        window=int(sma_choice), min_periods=1).mean().values
                if add_rsi:
                    # RSI using Wilder smoothing (EWMA approximation) with fixed 14
                    window_rsi = 14
                    delta = close_proc.diff().fillna(0)
                    up = delta.clip(lower=0)
                    down = -1 * delta.clip(upper=0)
                    roll_up = up.ewm(alpha=1.0/window_rsi, adjust=False).mean()
                    roll_down = down.ewm(
                        alpha=1.0/window_rsi, adjust=False).mean()
                    rs = roll_up / roll_down.replace(0, np.nan)
                    rsi = 100 - (100 / (1 + rs))
                    df_feats["rsi"] = rsi.fillna(0).values

                # (lag features removed)

                # Display full preview sorted by date (latest first) and offer full-download
                try:
                    # Try to sort by parsed dates so ordering is chronological
                    df_feats_display = df_feats.copy()
                    df_feats_display["_dt"] = pd.to_datetime(
                        df_feats_display["date"]).dt.tz_localize(None)
                    df_feats_display = df_feats_display.sort_values(
                        by="_dt", ascending=False).drop(columns=["_dt"]).reset_index(drop=True)
                except Exception:
                    df_feats_display = df_feats.sort_values(
                        by="date", ascending=False).reset_index(drop=True)

                st.subheader("Feature preview (all rows — latest first)")
                st.dataframe(df_feats_display)
                st.caption(
                    f"Rows: {len(df_feats_display)} • Columns: {', '.join(df_feats_display.columns.tolist())}")

                try:
                    csv_bytes = df_feats_display.to_csv(
                        index=False).encode("utf-8")
                    st.download_button(
                        label="Download preview CSV (full)",
                        data=csv_bytes,
                        file_name=f"{ticker}_preview_features.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception:
                    pass

                # Warn about redundant features when SMA window equals the rolling mean window
                try:
                    redundancy = []
                    rw_int = int(rolling_window)
                    if add_sma and int(sma_choice) > 0 and int(sma_choice) == rw_int:
                        redundancy.append(f"SMA (window={int(sma_choice)})")
                    if redundancy:
                        st.warning(
                            f"Redundant features detected: {', '.join(redundancy)} match the rolling mean window and may be highly correlated.")
                except Exception:
                    pass

                # Check correlation between rolling_mean and close when both exist
                try:
                    if 'rolling_mean' in df_feats.columns and 'close' in df_feats.columns:
                        corr_val = float(df_feats['close'].corr(
                            df_feats['rolling_mean']))
                        # Show a small metric and warn when extremely high correlation
                        st.metric("Close vs Rolling mean (Pearson r)",
                                  f"{corr_val:.3f}")
                        if abs(corr_val) >= 0.90:
                            st.warning(
                                f"The rolling mean and close are highly correlated (r={corr_val:.2f}). Consider dropping one to reduce redundancy.")
                except Exception:
                    pass

                # Correlation heatmap (colored) to show potential collinearity
                try:
                    # Use numeric-only columns to avoid non-numeric 'date' interfering
                    df_num = df_feats.select_dtypes(include=[np.number])
                    if df_num is not None and not df_num.empty and df_num.shape[1] >= 1:
                        corr = df_num.corr()
                        # Build a stable heatmap using graph_objects.Heatmap which
                        # avoids cases where px.imshow combined with _safe_plot
                        # may not render (image-style traces don't have .y/.x).
                        import plotly.graph_objects as _go
                        fig_corr = _go.Figure(data=_go.Heatmap(
                            z=corr.values,
                            x=corr.columns.tolist(),
                            y=corr.index.tolist(),
                            colorscale='RdBu',
                            zmin=-1, zmax=1,
                            colorbar=dict(title='Pearson r')
                        ))
                        # Add numeric annotations for small matrices for readability
                        if corr.shape[0] <= 10:
                            annotations = []
                            for i, row in enumerate(corr.index):
                                for j, col in enumerate(corr.columns):
                                    annotations.append(dict(
                                        x=col, y=row, text=f"{corr.iloc[i,j]:.2f}",
                                        showarrow=False, font=dict(color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")
                                    ))
                            fig_corr.update_layout(annotations=annotations)
                        fig_corr.update_layout(title="Feature correlation (Pearson)", height=420,
                                               template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
                        # Directly render the Plotly figure (heatmap traces are safe)
                        st.plotly_chart(fig_corr, use_container_width=True)
                except Exception:
                    pass

                # (ACF/PACF preview removed)

        # Normalization / Scaling
        st.subheader("Normalization / Scaling")
        scaler_options = {
            "Standard scaler": "standard",
            "MinMax scaling (0-1)": "minmax",
            "Robust scaling (median/IQR)": "robust",
        }
        scaler_label = st.selectbox("Choose normalization", options=list(scaler_options.keys(
        )), index=0, help="Pick how the model input is preprocessed and scaled. Default is Standard scaler.")
        scaler_choice = scaler_options.get(scaler_label, "standard")

    run_btn = st.button(
        "Run analysis",
        type="primary",
        disabled=(not is_valid) or (not dates_ok),
        help="Start training and forecasting with the selected settings. This may take several minutes depending on model and data size.",
    )

st.title("Stock Forecast")

# Fresh run trigger: capture inputs, clear state, rerun
if run_btn and ticker:
    params = {
        "ticker": ticker,
        # date range will be handled using session state selection below
        "model_kind": model_kind,
        "window": window,
        "horizon": horizon,
        # Use the selected horizon (1..14) as the multi-horizon forecast length
        "multi_horizon": int(horizon),
        "test_split": test_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout": dropout,
        "learning_rate": float(learning_rate),
        "filter_outliers": filter_outliers,
        "outlier_threshold": float(outlier_threshold),
        "soften_spikes": soften,
        "spike_threshold": float(spike_threshold),
        "spike_factor": float(spike_factor),
        "add_rolling_mean": bool(add_rolling_mean),
        "rolling_window": int(rolling_window),
        "sma_window": int(sma_choice) if add_sma else 0,
        "add_rsi": bool(add_rsi),
        "rsi_window": 14,
        "scaler": scaler_choice,
    }
    # Preserve existing selections (ticker, dates, etc.)
    if "analysis" in st.session_state:
        del st.session_state["analysis"]
    st.session_state["pending_run"] = params
    st.rerun()

# Execute pending run on a clean state
if st.session_state.get("pending_run"):
    p = st.session_state.pop("pending_run")
    try:
        with st.spinner("Fetching data…"):
            # Always fetch max, then slice by selected date range
            df_all = fetch_prices(p["ticker"], period="max")
        sdt = st.session_state.get("start_date")
        edt = st.session_state.get("end_date")
        if sdt and edt:
            start = pd.to_datetime(sdt)
            end = pd.to_datetime(edt)
            df = df_all[(pd.to_datetime(df_all["date"]).dt.tz_localize(None) >= start.tz_localize(None)) & (
                pd.to_datetime(df_all["date"]).dt.tz_localize(None) <= end.tz_localize(None))].reset_index(drop=True)
            if len(df) < max(30, int(p["window"]) + 5):
                st.warning(
                    "Selected date range is too short; consider a longer range for reliable training.")
        else:
            df = df_all
        name = get_company_name(p["ticker"]) or ""
        st.caption(f"{p['ticker']} • {name}")

        # Construct ModelConfig with core/required args first. Some
        # environments may have an older `ModelConfig` signature; avoid
        # passing optional kwargs that could raise TypeError. Set optional
        # attributes defensively after construction when supported.
        cfg = ModelConfig(
            window=p["window"],
            horizon=p.get("multi_horizon", p.get("horizon", 1)),
            test_split=p["test_split"],
            filter_outliers=p["filter_outliers"],
            outlier_threshold=float(p["outlier_threshold"]),
            soften_spikes=p["soften_spikes"],
            spike_threshold=float(p["spike_threshold"]),
            spike_factor=float(p["spike_factor"]),
            scaler=p.get("scaler", "standard"),
        )

        # Optional attributes: set only when the dataclass actually exposes them.
        optional_attrs = {
            "add_technical_features": False,
            "add_rolling_mean": bool(p.get("add_rolling_mean", False)),
            "rolling_window": int(p.get("rolling_window", 7)),
            "sma_window": int(p.get("sma_window", 0)),
            "add_rsi": bool(p.get("add_rsi", False)),
            "rsi_window": int(p.get("rsi_window", 14)),
        }
        for k, v in optional_attrs.items():
            try:
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            except Exception:
                # Be conservative: ignore any attribute errors to avoid
                # crashing the Streamlit UI when signatures differ.
                pass

        # Determine model kind: use sklearn as fallback when TensorFlow isn't available
        if not TF_AVAILABLE:
            mk = "sklearn"
        else:
            mk = p.get("model_kind") or "lstm"
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
            learning_rate=p.get("learning_rate", 1e-4),
            progress_callback=on_progress if mk != "sklearn" else None,
            callbacks={
                "early_stopping": p.get("enable_early_stopping", True),
                "model_checkpoint": p.get("enable_checkpoint", True),
                "force_retrain": p.get("force_retrain", False),
            },
        )
        progress.empty()
        status.empty()

        # Save fresh analysis
        st.session_state["analysis"] = {
            "ticker": p["ticker"],
            "period": "custom-range",
            "name": name,
            "df": df,
            "result": result,
            "params": {
                "test_split": p["test_split"],
                "epochs": p["epochs"],
                "batch_size": p["batch_size"],
                "dropout": p["dropout"],
                "learning_rate": p.get("learning_rate", 1e-4),
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
        train_idx_original = kept[: boundary] if boundary <= len(
            kept) else list(kept)
        training_df = df.iloc[train_idx_original].reset_index(drop=True)
    except Exception:
        training_df = df.copy().reset_index(drop=True)

    # Dates for plotting
    dates_series = pd.to_datetime(df["date"]).dt.tz_localize(None)
    dates = dates_series.dt.strftime("%Y-%m-%d").tolist()
    prices = df["price"].values.astype(float)

    # Build an array of scalar fitted values for plotting (use h1 when multi-horizon)
    fitted_on_original = [np.nan] * len(df)
    for j, idx in enumerate(kept):
        if j < len(fitted):
            val = fitted[j]
            # If multi-horizon, val may be a list/array; use first horizon for line plot
            if isinstance(val, (list, tuple, np.ndarray)):
                try:
                    fitted_on_original[idx] = float(
                        val[0]) if len(val) > 0 else np.nan
                except Exception:
                    fitted_on_original[idx] = np.nan
            else:
                try:
                    fitted_on_original[idx] = float(val)
                except Exception:
                    fitted_on_original[idx] = np.nan
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
    _add_trace_safe(fig, dates, prices, mode="lines",
                    name="Actual", line=dict(color="#000000", width=2))
    train_mask = [i <= test_start_original for i in range(len(df))]
    test_mask = [i > test_start_original for i in range(len(df))]
    _add_trace_safe(fig, dates, [fitted_on_original[i] if train_mask[i] else None for i in range(
        len(df))], mode="lines", name="Training fit", line=dict(color="#FFA500", width=2))
    _add_trace_safe(fig, dates, [fitted_on_original[i] if test_mask[i] else None for i in range(
        len(df))], mode="lines", name="Testing fit", line=dict(color="#2E8B57", width=2))
    # Ensure future values are numeric scalars for plotting (use first horizon if multi-step)
    future_plot = []
    try:
        for v in future:
            if isinstance(v, (list, tuple, np.ndarray)):
                future_plot.append(float(v[0]) if len(v) > 0 else float('nan'))
            else:
                future_plot.append(float(v))
    except Exception:
        # Fallback: try casting entire list
        try:
            future_plot = [float(x) for x in future]
        except Exception:
            future_plot = future
    _add_trace_safe(fig, future_dates, future_plot, mode="lines",
                    name="Future", line=dict(color="#FF0000", width=2, dash="dot"))
    fig.update_layout(template="plotly_white", height=550,
                      legend_orientation="h", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
    _safe_plot(fig)

    # Quick view range removed per request

    # Panels
    st.subheader("Used config")
    p = analysis["params"]
    # Show persisted feature columns and scaler when available
    feature_cols = result.get("feature_cols") if isinstance(
        result, dict) else None
    scaler_name = None
    fitted_scaler = result.get(
        "fitted_scaler") if isinstance(result, dict) else None
    try:
        if fitted_scaler is not None:
            scaler_name = type(fitted_scaler).__name__
    except Exception:
        scaler_name = result.get("scaler") if isinstance(
            result, dict) else None

    st.json({
        "features": feature_cols or "close",
        "scaler": scaler_name,
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
        # Top-line metrics (first horizon) and per-horizon breakdown
        cols = st.columns(4)
        cols[0].metric("RMSE (h1)", f"{m.get('RMSE', float('nan')):.2f}")
        cols[1].metric("MAE (h1)", f"{m.get('MAE', float('nan')):.2f}")
        cols[2].metric("MAPE (%) (h1)", f"{m.get('MAPE', float('nan')):.2f}")
        cols[3].metric("R² (h1)", f"{m.get('R2', float('nan')):.3f}")
        # Show per-horizon table if available
        per_h = m.get("per_horizon") if isinstance(m, dict) else None
        if per_h:
            rows = []
            for h, metrics in sorted(per_h.items(), key=lambda x: int(x[0][1:])):
                rows.append({"horizon": h, "RMSE": metrics.get("RMSE"), "MAE": metrics.get(
                    "MAE"), "MAPE": metrics.get("MAPE"), "R2": metrics.get("R2")})
            st.table(pd.DataFrame(rows))
        hist = result.get("history") or {}
        if hist and (hist.get("loss") or hist.get("val_loss")):
            ep = list(range(1, max(len(hist.get("loss", [])),
                      len(hist.get("val_loss", []))) + 1))
            figh = go.Figure()
            if hist.get("loss"):
                y_loss = list(hist["loss"])
                _add_trace_safe(figh, ep, y_loss, mode="lines",
                                name="loss", line=dict(color="#1f77b4"))
            if hist.get("val_loss"):
                y_vloss = list(hist["val_loss"])
                _add_trace_safe(figh, ep, y_vloss, mode="lines",
                                name="val_loss", line=dict(color="#ff7f0e"))
            figh.update_layout(
                height=250, template="plotly_white", title="Training history")
            _safe_plot(figh)
        # Show an unscaled window preview table: Date | past window values | Prediction
        try:
            window_size = int(result.get("window", 0))
            close_filtered = result.get("close_filtered") or []
            dates_filtered = result.get("dates_filtered") or []
            fitted_filtered = result.get("fitted_on_filtered") or []
            if window_size and len(close_filtered) >= window_size + 1:
                rows = []
                boundary = int(result.get(
                    "train_test_boundary", len(close_filtered)))
                start_i = max(window_size, boundary - 5)
                end_i = min(boundary, len(close_filtered))
                # Determine selected forecast length
                fh = int(result.get("horizon_used", p.get(
                    "multi_horizon", p.get("horizon", 1))))
                pred_cols = [f"pred_t+{k+1}" for k in range(fh)]
                for i in range(start_i, end_i):
                    date_str = pd.to_datetime(dates_filtered[i]).strftime(
                        "%Y-%m-%d") if i < len(dates_filtered) else str(i)
                    window_vals = close_filtered[i - window_size: i]
                    pred_vals = fitted_filtered[i] if i < len(
                        fitted_filtered) else None
                    # pred_vals may be list-like for multi-horizon
                    if isinstance(pred_vals, list) or hasattr(pred_vals, '__iter__'):
                        pvals = list(pred_vals)[:fh]
                    else:
                        pvals = [pred_vals]
                    rows.append([date_str] + list(window_vals) + pvals)
                cols = ["Date"] + \
                    [f"t-{k}" for k in range(window_size, 0, -1)] + pred_cols
                st.dataframe(pd.DataFrame(rows, columns=cols))
        except Exception:
            pass
            # Removed: training/plotted CSV downloads and preview tables per user request

else:
    # When a ticker is selected but before running, show the raw price chart.
    selected = st.session_state.get("ticker", None)
    if selected and validate_ticker_symbol(selected):
        with st.spinner("Fetching data…"):
            df_all = fetch_prices(selected, period="max")
        st.subheader(f"{selected} price history")
        dates_series = pd.to_datetime(df_all["date"]).dt.tz_localize(None)
        dates = dates_series.dt.strftime("%Y-%m-%d").tolist()
        prices = df_all["price"].astype(float).values
        fig0 = go.Figure()
        _add_trace_safe(fig0, dates, prices, mode="lines",
                        name="Actual", line=dict(color="#000000", width=2))
        fig0.update_layout(template="plotly_white",
                           height=450, legend_orientation="h")
        _safe_plot(fig0)
    # Guidance is provided via widget tooltips; no extra caption needed.
