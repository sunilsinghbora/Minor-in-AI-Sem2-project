# Stock Forecast (Streamlit)

A Python-only Streamlit app for quick stock analysis and forecasting with:
- Close-only model (StandardScaler + sliding windows; fast scikit-learn regressor by default)
- Optional inclusive outlier removal on absolute day-over-day % change
- Optional train-only spike softening (threshold and factor)
- Chronological train/test split
- Plotly chart: Actual, Training fit, Testing fit, and Future forecast
- Company name next to ticker

Neural models (LSTM/GRU variants) are supported and enabled on Streamlit Cloud via a CPU build of TensorFlow.

## Quick start

- Install deps
  - pip install -r requirements.txt
- Run
  - streamlit run streamlit_app.py

## Neural models

This app uses TensorFlow (CPU) 2.20.0 to enable LSTM/GRU on Cloud. First deploy after this change will be slower. If builds ever fail due to resource limits, you can remove TensorFlow from `requirements.txt` to fall back to sklearn-only.

Local setup tip: if you’re running locally, you can install a compatible TF version too:

```
pip install tensorflow==2.20.0
```

Without TensorFlow, the app automatically hides neural options and uses the fast sklearn model.

## Notes
- Defaults are chosen to run everywhere (CPU). Neural models are heavier and optional.
- Network calls go to Yahoo for prices; if blocked, try later.
