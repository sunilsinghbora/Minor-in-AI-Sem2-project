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

Note: Installing TensorFlow on Streamlit Cloud may increase deploy times or fail on resource-limited plans. To enable neural models, install TensorFlow locally or uncomment the `tensorflow-cpu` line in `requirements.txt` and redeploy on a plan that supports larger builds.

## Notes
- Defaults are chosen to run everywhere (CPU). Neural models are heavier and optional.
- Network calls go to Yahoo for prices; if blocked, try later.


## Disclaimer

The information and predictions provided by this app are for educational and informational purposes only and should not be considered as financial, investment, or trading advice.

The stock market is inherently volatile and unpredictable. While we use historical data and machine learning models to generate forecasts, past performance is not indicative of future results.

You are solely responsible for any investment decisions you make based on the information provided by this app. We strongly recommend consulting with a licensed financial advisor before making any investment decisions.

The creators of this app are not liable for any losses or damages arising from the use of this app or its content.
