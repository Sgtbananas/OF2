import itertools
import pandas as pd
import joblib
import os
from core.signals import generate_signals
from core.trade import backtest_strategy
from core.features import add_all_features  # Ensure feature engineering is applied

PIPELINE_PATH = "ml_filter_pipeline.pkl"

def load_pipeline():
    if os.path.exists(PIPELINE_PATH):
        return joblib.load(PIPELINE_PATH)
    else:
        raise FileNotFoundError(f"[ERROR] Model pipeline file not found: {PIPELINE_PATH}")

def align_input_features(df, expected_features):
    """
    Ensures that df has all expected columns in correct order,
    filling missing columns with 0.
    """
    aligned = df.copy()
    for col in expected_features:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[expected_features]

def optimize_strategies(df, all_strategies, config, ml_filter=None):
    results = []
    
    # Apply feature engineering
    df = add_all_features(df)

    # Load full pipeline
    pipeline = load_pipeline()

    # Determine input feature names the pipeline expects
    input_features = pipeline.named_steps["selector"].estimator_.feature_importances_.shape[0]
    scaler_input_order = pipeline.named_steps["scaler"].get_feature_names_out(
        input_features if isinstance(input_features, list) else None
    ) if hasattr(pipeline.named_steps["scaler"], "get_feature_names_out") else df.columns

    aligned_df = align_input_features(df, scaler_input_order)
    ml_features = pipeline.transform(aligned_df)

    for r in range(1, len(all_strategies) + 1):
        for strat_combo in itertools.combinations(all_strategies, r):
            test_config = config.copy()
            test_config["strategies"] = list(strat_combo)

            signal = generate_signals(df, test_config)

            trades = backtest_strategy(
                df,
                signal,
                test_config,
                config.get("symbol", "TEST"),
                ml_filter=ml_filter,
                ml_features=ml_features  # Already aligned + transformed
            )

            if trades:
                pnl = sum(t.get("pnl", 0) for t in trades if "pnl" in t)
                wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
                losses = sum(1 for t in trades if t.get("pnl", 0) <= 0)
                win_rate = wins / (wins + losses + 1e-9)
                results.append({
                    "strategies": strat_combo,
                    "pnl": pnl,
                    "win_rate": round(win_rate, 2),
                    "trades": len(trades)
                })

    result_df = pd.DataFrame(results).sort_values(by="pnl", ascending=False)
    print("[OPTIMIZER] Top strategy combos:")
    print(result_df.head(10))
    return result_df
