import os
import time
import subprocess
import itertools
import joblib
import pandas as pd
from core.signals import generate_signals
from core.trade import backtest_strategy
from core.features import add_all_features

PIPELINE_PATH = "ml_filter_pipeline.pkl"
TRAIN_SCRIPT = "train_ml_filter.py"
MODEL_MAX_AGE_HOURS = 24

def is_pipeline_fresh():
    if not os.path.exists(PIPELINE_PATH):
        return False
    age_seconds = time.time() - os.path.getmtime(PIPELINE_PATH)
    return age_seconds < MODEL_MAX_AGE_HOURS * 3600

def ensure_pipeline():
    """Ensure a fresh ML pipeline exists, or retrain it. Then return the loaded pipeline."""
    if not is_pipeline_fresh():
        print("[OPTIMIZER] Pipeline missing or stale. Triggering training...")
        result = subprocess.run(["python", TRAIN_SCRIPT])
        if result.returncode != 0 or not os.path.exists(PIPELINE_PATH):
            raise RuntimeError("[OPTIMIZER] Training failed or pipeline not saved.")
    else:
        print(f"[OPTIMIZER] Using fresh ML pipeline from: {PIPELINE_PATH}")

    loaded = joblib.load(PIPELINE_PATH)
    if isinstance(loaded, dict):
        model_name = type(loaded['model']).__name__
        print(f"[OPTIMIZER] Active ML model: {model_name}")
    elif hasattr(loaded, 'named_steps'):
        model_name = type(loaded.named_steps['classifier']).__name__
        print(f"[OPTIMIZER] Active ML model: {model_name}")
    return loaded

def align_features(df, pipeline):
    """Align dataframe with expected pipeline input feature structure."""
    # Always get the full feature list
    if isinstance(pipeline, dict):
        feature_list = pipeline.get('full_feature_list', df.columns)
    else:
        # fallback for legacy pipeline types
        feature_list = df.columns
    aligned = df.copy()
    for col in feature_list:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[feature_list]

def optimize_strategies(df, all_strategies, config, ml_filter=True):
    ensure_pipeline()
    pipeline = joblib.load(PIPELINE_PATH)

    df = add_all_features(df)
    aligned_df = align_features(df, pipeline)
    # If using dict, manually apply scaler/selector/model if needed (not shown), else use pipeline.transform
    if hasattr(pipeline, "transform"):
        transformed_features = pipeline.transform(aligned_df)
    else:
        scaler = pipeline['scaler']
        selector = pipeline['selector']
        X_scaled = scaler.transform(aligned_df)
        transformed_features = selector.transform(X_scaled)

    results = []

    for r in range(1, len(all_strategies) + 1):
        for combo in itertools.combinations(all_strategies, r):
            test_config = config.copy()
            test_config["strategies"] = list(combo)

            signal = generate_signals(df, test_config)

            trades = backtest_strategy(
                df,
                signal,
                test_config,
                config.get("symbol", "ASSET"),
                ml_filter=ml_filter,
                ml_features=transformed_features
            )

            if trades:
                pnl = sum(t["pnl"] for t in trades if "pnl" in t)
                wins = sum(1 for t in trades if t["pnl"] > 0)
                losses = sum(1 for t in trades if t["pnl"] <= 0)
                win_rate = wins / (wins + losses + 1e-9)
                results.append({
                    "strategies": combo,
                    "pnl": round(pnl, 2),
                    "win_rate": round(win_rate, 2),
                    "trades": len(trades)
                })

    results_df = pd.DataFrame(results).sort_values(by="pnl", ascending=False)
    print("[OPTIMIZER] Top strategies:")
    print(results_df.head(10))
    return results_df