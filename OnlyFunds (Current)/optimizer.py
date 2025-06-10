import os
import subprocess
import itertools
import joblib
import pandas as pd
from core.signals import generate_signals
from core.trade import backtest_strategy
from core.features import add_all_features

PIPELINE_PATH = "ml_filter_pipeline.pkl"
TRAIN_SCRIPT = "train_ml_filter.py"

def ensure_pipeline():
    """Ensure the ML pipeline exists, or retrain it automatically."""
    if not os.path.exists(PIPELINE_PATH):
        print("[OPTIMIZER] Pipeline not found. Auto-training now...")
        result = subprocess.run(["python", TRAIN_SCRIPT])
        if result.returncode != 0 or not os.path.exists(PIPELINE_PATH):
            raise RuntimeError("[OPTIMIZER] Training failed or pipeline not created.")
    else:
        print("[OPTIMIZER] Found existing ML pipeline.")

def align_features(df, pipeline):
    """Align live dataframe with pipeline expectations."""
    scaler = pipeline.named_steps["scaler"]
    expected_features = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else df.columns
    aligned = df.copy()
    for col in expected_features:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[expected_features]

def optimize_strategies(df, all_strategies, config, ml_filter=True):
    ensure_pipeline()
    pipeline = joblib.load(PIPELINE_PATH)

    df = add_all_features(df)
    aligned_df = align_features(df, pipeline)
    transformed = pipeline.transform(aligned_df)

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
                config.get("symbol", "SYMBOL"),
                ml_filter=ml_filter,
                ml_features=transformed
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

    df_results = pd.DataFrame(results).sort_values(by="pnl", ascending=False)
    print("[OPTIMIZER] Top strategy sets:")
    print(df_results.head(10))
    return df_results
