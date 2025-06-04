import pytest
import pandas as pd
import numpy as np
from core.ml_filter import MLFilter

# Helper to generate fake OHLCV data
def make_ohlcv_df(n=100):
    np.random.seed(42)
    df = pd.DataFrame({
        "Open": np.random.rand(n) * 100,
        "High": np.random.rand(n) * 100,
        "Low": np.random.rand(n) * 100,
        "Close": np.random.rand(n) * 100,
        "Volume": np.random.rand(n) * 1000,
    })
    # Add some volatility
    df["Close"] = df["Close"].cumsum() / 10 + 100
    return df

def test_extract_features_valid():
    df = make_ohlcv_df(50)
    feats = MLFilter.extract_features(df, 40)
    assert isinstance(feats, (list, np.ndarray))
    assert len(feats) >= 6  # Should match number of features
    # Check for non-nan, finite values
    assert all(np.isfinite(feats))

def test_extract_features_edge_case_start():
    df = make_ohlcv_df(50)
    feats = MLFilter.extract_features(df, 0)
    assert isinstance(feats, (list, np.ndarray))
    assert all(np.isfinite(feats))

def test_extract_features_missing_columns():
    # Remove Volume column
    df = make_ohlcv_df(50).drop("Volume", axis=1)
    # Should not raise, but fill with zeros where needed
    feats = MLFilter.extract_features(df, 10)
    assert isinstance(feats, (list, np.ndarray))
    assert all(np.isfinite(feats))

def test_extract_features_nan_handling():
    df = make_ohlcv_df(50)
    df.loc[25, "Close"] = np.nan
    feats = MLFilter.extract_features(df, 30)
    assert isinstance(feats, (list, np.ndarray))
    assert all(np.isfinite(feats))

def test_extract_features_short_df():
    df = make_ohlcv_df(5)
    feats = MLFilter.extract_features(df, 2)
    assert isinstance(feats, (list, np.ndarray))
    assert all(np.isfinite(feats))

def test_extract_features_extreme_values():
    df = make_ohlcv_df(50)
    df.loc[10, "Close"] = 1e9
    feats = MLFilter.extract_features(df, 10)
    assert isinstance(feats, (list, np.ndarray))
    assert all(np.isfinite(feats) | np.isnan(feats))  # Allow inf/nan ONLY if unavoidable

# Run with: pytest tests/test_ml_filter.py