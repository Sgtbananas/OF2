import pandas as pd
import numpy as np
from core.data_handler import fetch_ohlcv
from core.simulator import simulate_trades
from core.strategy_loader import load_strategy
from core.order_manager import execute_trade, dry_run_trade, log_backtest
from core.logger import log_message
from core.visuals import print_summary
from core.ml_filter import MLFilter
from core.features import add_all_features
import os

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

        df = add_all_features(df)

        # === BULLETPROOF PATCH: Build df_ml for MLFilter, with only required features and order ===
        df_ml = None
        if ml_filter is not None and hasattr(ml_filter, "features") and ml_filter.features:
            for col in ml_filter.features:
                if col not in df.columns:
                    df[col] = np.nan  # or 0.0 if you prefer
            df_ml = df[ml_filter.features]
            print("[DEBUG][LIVE] MLFilter expects features:", ml_filter.features)
            print("[DEBUG][LIVE] DataFrame columns for MLFilter:", list(df_ml.columns))
        # else: strategies get the full df, MLFilter gets None

        strategies = {}
        for strat_name in all_strats:
            try:
                strategy = load_strategy(strat_name)
                signals = strategy.generate_signals(df)
                # Pass both the full df (for strategies) and df_ml (for MLFilter)
                result = simulate_trades(
                    df,
                    signals,
                    symbol,
                    target,
                    ml_filter=ml_filter,
                    ml_features=df_ml
                )
                strategies[strat_name] = result
            except Exception as e:
                log_message(f"⚠️ Strategy {strat_name} failed: {e}")

        if not strategies:
            log_message(f"❌ No successful strategies for {symbol}. Skipping.")
            continue

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