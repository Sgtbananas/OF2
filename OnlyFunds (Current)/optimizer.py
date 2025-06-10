import itertools
import pandas as pd
import joblib
import os

from core.signals import generate_signals
from core.trade import backtest_strategy

SELECTOR_PATH = "feature_selector.pkl"

def get_selector_and_features():
    if os.path.exists(SELECTOR_PATH):
        fs_data = joblib.load(SELECTOR_PATH)
        return (
            fs_data["model"],
            fs_data["selected_features"],
            fs_data["feature_mask"],
            fs_data["full_feature_list"]
        )
    return None, None, None, None

def align_features_for_ml(df, full_feature_list):
    aligned = df.copy()
    for col in full_feature_list:
        if col not in aligned.columns:
            aligned[col] = 0.0
    aligned = aligned[full_feature_list]
    return aligned

def optimize_strategies(df, all_strategies, config, ml_filter=None):
    results = []
    selector, selected_features, feature_mask, full_feature_list = get_selector_and_features()

    ml_features_df = None
    if ml_filter is not None and full_feature_list is not None and selector is not None:
        aligned_df = align_features_for_ml(df, full_feature_list)

        # Defensive assertion
        assert aligned_df.shape[1] == len(feature_mask), (
            f"[ERROR] Feature mismatch: aligned_df has {aligned_df.shape[1]}, mask expects {len(feature_mask)}"
        )

        ml_features_df = selector.transform(aligned_df)

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
                ml_features=ml_features_df
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
    return result_df
