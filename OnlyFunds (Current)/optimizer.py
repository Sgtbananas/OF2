import itertools
import pandas as pd
from core.signals import generate_signals
from core.trade import backtest_strategy
import joblib
import os

SELECTOR_PATH = "feature_selector.pkl"

def get_selector_and_features():
    if os.path.exists(SELECTOR_PATH):
        fs_data = joblib.load(SELECTOR_PATH)
        selector = fs_data["model"]
        selected_features = fs_data["selected_features"]
        feature_mask = fs_data["feature_mask"]
        return selector, selected_features, feature_mask
    return None, None, None

def optimize_strategies(df, all_strategies, config, ml_filter=None):
    results = []
    # PATCH: Robust feature alignment for ML inference
    selector, selected_features, feature_mask = get_selector_and_features()
    ml_features = None
    if ml_filter is not None and selected_features is not None:
        for col in selected_features:
            if col not in df.columns:
                df[col] = 0  # or np.nan or other default
        ml_features = df[selected_features]
        assert ml_features.shape[1] == len(feature_mask), \
            f"Feature mismatch: ml_features has {ml_features.shape[1]} cols, mask has {len(feature_mask)}"

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
                ml_features=ml_features
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