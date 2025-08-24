# Stock Forecast (Streamlit)

A Python-only Streamlit app for quick stock analysis and forecasting with:
- Close-only model (StandardScaler + sliding windows; fast scikit-learn regressor by default)
- Optional inclusive outlier removal on absolute day-over-day % change
- Optional train-only spike softening (threshold and factor)
- Chronological train/test split
- Plotly chart: Actual, Training fit, Testing fit, and Future forecast
- Company name next to ticker

Neural models (LSTM/GRU variants) are supported optionally if TensorFlow/Keras is installed. On Streamlit Cloud, we disable TensorFlow to keep installs fast and reliable.

## Quick start

- Install deps
  - pip install -r requirements.txt
- Run
  - streamlit run streamlit_app.py

## Optional: enable neural models locally

TensorFlow is not included in `requirements.txt` to avoid build failures on Python 3.13 in hosted environments. Locally, you can enable the neural models by installing a compatible TensorFlow build for your Python version, for example:

```
# Example (adjust to a TF version supported by your Python):
pip install tensorflow==2.20.0
```

If TensorFlow is not installed, the app will automatically fall back to the fast scikit-learn model and hide neural options in the UI.

## Notes
- Defaults are chosen to run everywhere (CPU). Neural models are heavier and optional.
- Network calls go to Yahoo for prices; if blocked, try later.
