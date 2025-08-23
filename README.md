# Stock Forecast (Streamlit)

A Python-only Streamlit app for quick stock analysis and forecasting with:
- Close-only model (StandardScaler + simple LSTM-like windowing but using fast sklearn regressor by default)
- Optional inclusive outlier removal on absolute day-over-day % change
- Optional train-only spike softening (threshold and factor)
- Chronological 90/10 train/test split
- Plotly chart: Actual (white), Training fit (orange), Testing fit (yellow), Future (green)
- Company name next to ticker
- News sentiment from RSS (only if >=5 articles in last ~90 days)
- History (training/validation loss) is simulated as sklearn doesn’t expose epochs; if you switch to Keras, the modal will show real curves

## Quick start

- Install deps
  - pip install -r requirements.txt
- Run
  - streamlit run streamlit_app.py

## Notes
- Defaults are chosen to run everywhere (CPU). For better accuracy, swap the regressor with a small Keras LSTM and enable GPU if available.
- Network calls go to Yahoo for prices and RSS; if blocked, try later.
