import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import numpy as np
import pandas as pd
from core.ml_filter import MLFilter

PIPELINE_PATH = "ml_filter_pipeline.pkl"

def test_pipeline_alignment():
    print("=== Bulletproof MLFilter End-to-End Alignment Test ===")
    p = joblib.load(PIPELINE_PATH)
    # Check pipeline structure
    assert 'model' in p and 'selector' in p and 'full_feature_list' in p, "Pipeline missing required keys"
    full_feature_list = p['full_feature_list']
    print(f"Full feature list ({len(full_feature_list)}): {full_feature_list}")
    print(f"Selector support mask shape: {p['selector'].get_support().shape}")

    # Create a DataFrame with all features, in correct order, with dummy values
    test_data = {f: np.random.randn(1)[0] for f in full_feature_list}
    # Optionally, remove a feature to test bulletproofing
    # test_data.pop(full_feature_list[0])

    df = pd.DataFrame([test_data])
    print("Test DataFrame columns:", df.columns.tolist())
    print("Test DataFrame shape:", df.shape)

    mlfilter = MLFilter(model_path=PIPELINE_PATH)
    try:
        feats = mlfilter.extract_features(df, 0)
        print("Extracted features shape after selector:", feats.shape)
        print("Test PASSED 🚀 — No axis/shape mismatch.")
    except Exception as e:
        print("Test FAILED ❌:", e)
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline_alignment()