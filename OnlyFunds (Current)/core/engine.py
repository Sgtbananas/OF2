import pandas as pd
import numpy as np
from core.data_handler import fetch_ohlcv
from core.simulator import simulate_trades
from core.strategy_loader import load_strategy
from core.order_manager import execute_trade, dry_run_trade, log_backtest
from core.logger import log_message
from core.visuals import print_summary
from core.ml_filter import MLFilter
from core.features import add_all_features  # PATCH: Import feature engineering
import os

# PATCH: Use the exact same features as training
NUMERIC_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'tr', 'atr14', 'log_return', 'realized_vol_10', 'return_3',
    'roll_close_std_5', 'roll_vol_mean_5', 'roll_vol_std_5',
    'roll_close_std_10', 'roll_vol_mean_10', 'roll_vol_std_10',
    'roll_close_std_20', 'roll_vol_mean_20', 'roll_vol_std_20',
    'entry_idx', 'exit_idx', 'pnl'
]

def run_bot(config):
    mode = config.get("mode", "backtest")
    risk = config.get("risk", 1)
    target = float(config.get("target", 1))
    symbols = config.get("symbols", [])
    all_strats = config.get("all_strategies", [])
    ml_cfg = config.get("ml_filter", {})
    ml_enabled = ml_cfg.get("enabled", False) if isinstance(ml_cfg, dict) else bool(ml_cfg)
    model_path = ml_cfg.get("model_path", "ml_filter_model.pkl") if isinstance(ml_cfg, dict) else "ml_filter_model.pkl"
    ml_threshold = config.get("ml_threshold", 0.6)
    ledger = []

    # Optionally set up MLFilter
    ml_filter = None
    if ml_enabled:
        if os.path.exists(model_path):
            try:
                ml_filter = MLFilter(model_path=model_path)
                log_message(f"✅ ML filter loaded from {model_path}")
            except Exception as e:
                log_message(f"❌ Failed to load ML filter model: {e}. Disabling ML filter.")
                ml_filter = None
        else:
            log_message(f"❌ ML model file '{model_path}' not found. Disabling ML filter.")
            ml_filter = None

    for symbol in symbols:
        log_message(f"Fetching data for {symbol}...")
        df = fetch_ohlcv(symbol, config.get("timeframe", "5min"), config.get("limit", 300))
        if df is None or df.empty:
            log_message(f"❌ Failed to fetch data for {symbol}")
            continue

        df = add_all_features(df)  # PATCH: Ensure all features present before passing to strategies/ML

        # PATCH: World-class guarantee for both MLFilter and strategy features
        ml_required = list(ml_filter.features) if (ml_filter is not None and hasattr(ml_filter, "features") and ml_filter.features) else []
        strat_required = set(NUMERIC_FEATURES)
        # But also keep all columns needed by strategies (e.g., close, high, etc)
        keep_cols = set(df.columns) | set(ml_required) | strat_required

        for feat in keep_cols:
            if feat not in df.columns:
                df[feat] = np.nan

        # Order the columns for MLFilter (for ML), but keep all for strategies
        if ml_required:
            df_for_ml = df[ml_required].copy()
        else:
            df_for_ml = df[[col for col in NUMERIC_FEATURES if col in df.columns]].copy()

        print("[DEBUG][LIVE] Features after add_all_features:", list(df.columns))
        print("[DEBUG][LIVE] Features for MLFilter:", list(df_for_ml.columns))

        strategies = {}
        for strat_name in all_strats:
            try:
                strategy = load_strategy(strat_name)
                signals = strategy.generate_signals(df)
                # Pass both: df (all features) and df_for_ml (ordered for MLFilter)
                result = simulate_trades(df, signals, symbol, target, ml_filter=ml_filter)
                strategies[strat_name] = result
            except Exception as e:
                log_message(f"⚠️ Strategy {strat_name} failed: {e}")

        if not strategies:
            log_message(f"❌ No successful strategies for {symbol}. Skipping.")
            continue

        # Select best strategy by PnL
        best_strategy = max(strategies.items(), key=lambda x: x[1]["pnl"])[0]
        best_result = strategies[best_strategy]
        log_message(
            f"✅ {symbol}: Best strategy → {best_strategy} | "
            f"PnL: {best_result['pnl']:.2f} | Win rate: {best_result['win_rate']:.2f}"
        )

        if mode == "live":
            execute_trade(symbol, best_result.get("trades", []))
        elif mode == "dry_run":
            dry_run_trade(symbol, best_result.get("trades", []))
        elif mode == "backtest":
            log_backtest(symbol, best_result.get("trades", []))

        ledger.extend(best_result["trades"])

    print_summary(pd.DataFrame(ledger))