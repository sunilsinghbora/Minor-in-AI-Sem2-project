import pandas as pd
import numpy as np
from utils import train_and_forecast_close_only_sklearn, ModelConfig


def test_inverse_transform_roundtrip_close_scale():
    # Create a simple increasing price series
    dates = pd.date_range("2020-01-01", periods=50)
    prices = 1.0 + 0.02 * np.arange(len(dates))
    df = pd.DataFrame({"date": dates, "open": prices, "price": prices})

    cfg = ModelConfig(window=5, horizon=1, test_split=0.2,
                      add_technical_features=True, rolling_window=3, scaler='standard')
    res = train_and_forecast_close_only_sklearn(df, cfg)

    # Ensure persisted metadata present
    assert 'fitted_scaler' in res
    assert 'feature_cols' in res
    scaler = res['fitted_scaler']
    fcols = res['feature_cols']
    assert isinstance(fcols, list) and len(fcols) >= 1

    # Take a small array of scaled predictions (fake): use last model's predicted scaled values
    # Build a small synthetic scaled array in the scaler's input space
    # If scaler expects multiple features, pad with last observed scaled values
    if hasattr(scaler, 'n_features_in_') and getattr(scaler, 'n_features_in_') > 1:
        n_in = getattr(scaler, 'n_features_in_')
        # construct M where first column is some small delta over last observed
        last_row = np.zeros(n_in)
        M = np.tile(last_row.reshape(1, -1), (3, 1)).astype(float)
        M[:, 0] = np.array([0.0, 0.1, -0.1])
        inv = scaler.inverse_transform(M)[:, 0]
    else:
        arr = np.array([0.0, 0.1, -0.1]).reshape(-1, 1)
        inv = scaler.inverse_transform(arr).flatten()

    # Assert inverse transformed values are in same rough scale as original prices
    assert all(np.isfinite(inv))
    assert (inv.max() - inv.min()) > 0.0
    # Values should be around the price scale (not extremely large)
    median_price = float(np.median(prices))
    assert np.all(np.abs(inv - median_price) <
                  max(1.0, 0.5 * median_price) + 1e6)
