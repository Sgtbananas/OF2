import pandas as pd
from datetime import datetime
import csv
import os
import traceback

def log_trade_sample(logfile, row, header=None, verbose=True):
    """Append a trade sample (features + label) to the given CSV file."""
    try:
        file_exists = os.path.isfile(logfile)
        with open(logfile, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header or row.keys())
            if not file_exists or os.stat(logfile).st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        if verbose:
            print(f"[ML_LOGGER] Logged trade sample for {row.get('symbol', 'UNKNOWN')} at {row.get('timestamp', '')}")
    except Exception as e:
        print(f"[ML_LOGGER] Error logging trade sample: {e}")
        traceback.print_exc()

def detect_market_regime(df, i, windows=[20, 50, 100]):
    """Detects simple market regimes (bull, bear, side) using moving averages."""
    regime = "unknown"
    try:
        if i >= max(windows):
            close = df["close"]
            ma_short = close.iloc[i-windows[0]+1:i+1].mean()
            ma_med = close.iloc[i-windows[1]+1:i+1].mean()
            ma_long = close.iloc[i-windows[2]+1:i+1].mean()
            if ma_short > ma_med > ma_long:
                regime = "bull"
            elif ma_short < ma_med < ma_long:
                regime = "bear"
            else:
                regime = "side"
    except Exception:
        pass
    return regime

def compute_market_features(df, i):
    """Extract advanced market context features: ATR, volatility, returns, etc."""
    features = {}
    # ATR (Average True Range)
    try:
        if i >= 14:
            high = df["high"].iloc[i-13:i+1]
            low = df["low"].iloc[i-13:i+1]
            close_prev = df["close"].iloc[i-14:i]
            tr = pd.concat([
                (high - low).abs(),
                (high - close_prev).abs(),
                (low - close_prev).abs()
            ], axis=1).max(axis=1)
            features["atr14"] = tr.mean()
        else:
            features["atr14"] = None
    except Exception:
        features["atr14"] = None
    # Realized volatility (returns std over 10 bars)
    try:
        if i >= 10:
            returns = df["close"].pct_change().iloc[i-9:i+1]
            features["realized_vol_10"] = returns.std()
        else:
            features["realized_vol_10"] = None
    except Exception:
        features["realized_vol_10"] = None
    # Recent return
    try:
        if i >= 3:
            features["return_3"] = (df["close"].iloc[i] / df["close"].iloc[i-3]) - 1
        else:
            features["return_3"] = None
    except Exception:
        features["return_3"] = None
    return features

def enrich_features(
    df, i, symbol, entry_idx, entry_time, entry_price, exit_price, pnl, reason, signal,
    hyperparams=None
):
    """World-class feature engineering: add rolling stats, volatility, regime, hyperparams."""
    row = {}
    # Basic fields
    row.update({
        "symbol": symbol,
        "timestamp": entry_time,
        "entry_idx": entry_idx,
        "exit_idx": i,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "reason": reason,
        "signal": signal,
        "close": df.iloc[i].get("close", None),
        "high": df.iloc[i].get("high", None),
        "low": df.iloc[i].get("low", None),
        "volume": df.iloc[i].get("volume", None),
    })
    # Add all available indicators/features from the DataFrame
    for feat in df.columns:
        if feat not in row:
            row[feat] = df.iloc[i].get(feat, None)
    # Rolling mean/volatility (last 5, 10, 20 bars)
    for window in [5, 10, 20]:
        if i >= window:
            row[f"roll_close_mean_{window}"] = df["close"].iloc[i-window+1:i+1].mean()
            row[f"roll_close_std_{window}"] = df["close"].iloc[i-window+1:i+1].std()
            row[f"roll_vol_mean_{window}"] = df["volume"].iloc[i-window+1:i+1].mean()
            row[f"roll_vol_std_{window}"] = df["volume"].iloc[i-window+1:i+1].std()
        else:
            row[f"roll_close_mean_{window}"] = None
            row[f"roll_close_std_{window}"] = None
            row[f"roll_vol_mean_{window}"] = None
            row[f"roll_vol_std_{window}"] = None
    # Time of day, day of week
    try:
        dt = pd.to_datetime(row["timestamp"])
        row["hour_of_day"] = dt.hour
        row["day_of_week"] = dt.dayofweek
    except Exception:
        row["hour_of_day"] = None
        row["day_of_week"] = None
    # Market regime
    row["market_regime"] = detect_market_regime(df, i)
    # Market features (volatility, ATR, recent return)
    row.update(compute_market_features(df, i))
    # Hyperparameters (if present in config)
    if hyperparams:
        for k, v in hyperparams.items():
            row[f"hp_{k}"] = v
    # Label: 1 if win, 0 otherwise
    row["label"] = 1 if pnl > 0 else 0
    return row

def simulate_trades(
    df, signals, symbol, target, ml_filter=None,
    stop_loss=None, trailing_stop=None,
    log_ml_data: bool = True,  # Always log ML data unless explicitly disabled
    ml_logfile: str = "ml_training_data.csv",
    verbose_ml_log: bool = False,
    hyperparams: dict = None   # Pass hyperparameters for logging
):
    """
    Simulates trades based on provided signals and optional ML filtering and risk management.
    Optionally logs trade data for ML training.
    Args:
        df (pd.DataFrame): DataFrame with price data.
        signals (pd.Series): Series with 1 (buy), -1 (sell), 0 (hold) signals.
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        target (float): Target profit/loss per trade.
        ml_filter (optional): ML filter object for filtering trades.
        stop_loss (float, optional): Stop loss as a fraction (e.g. 0.02 for 2%).
        trailing_stop (float, optional): Trailing stop as a fraction (e.g. 0.01 for 1%).
        log_ml_data (bool): If True, log trade data for ML training.
        ml_logfile (str): Path to the CSV file for logging trade data.
        verbose_ml_log (bool): Print every ML log row for debug.
        hyperparams (dict): Hyperparameters used for the strategy (for logging).

    Returns:
        dict: Contains trades (list of dicts), total PnL, and win rate.
    """
    trades = []
    position = None
    entry_price = 0.0
    highest_price = 0.0  # For trailing stop
    cumulative_pnl = 0.0
    win_count = 0

    # For ML logging: collect all possible feature columns (including new engineered ones)
    base_features = [
        "symbol", "timestamp", "entry_idx", "exit_idx",
        "entry_price", "exit_price", "pnl", "reason",
        "signal", "close", "high", "low", "volume",
        "hour_of_day", "day_of_week", "market_regime",
        "atr14", "realized_vol_10", "return_3"
    ]
    # Add all indicator columns from df if present (excluding duplicates)
    ml_feature_fields = base_features + [
        col for col in df.columns if col not in base_features
    ]
    # Add engineered rolling features and hyperparameters
    for window in [5, 10, 20]:
        ml_feature_fields += [
            f"roll_close_mean_{window}", f"roll_close_std_{window}",
            f"roll_vol_mean_{window}", f"roll_vol_std_{window}"
        ]
    # Add hyperparameter fields if present
    if hyperparams is not None:
        ml_feature_fields += [f"hp_{k}" for k in hyperparams.keys()]
    ml_feature_fields += ["label"]

    for i in range(len(df)):
        price = df.iloc[i]["close"]
        signal = signals.iloc[i]

        # Entry signal with ML filter
        if position is None and signal == 1:
            should_enter = True
            if ml_filter is not None and hasattr(ml_filter, "should_enter"):
                should_enter = ml_filter.should_enter(df, i, signal)
            if should_enter:
                position = "long"
                entry_price = price
                highest_price = price
                entry_idx = i
                entry_row = df.iloc[i]
                entry_time = entry_row.get("timestamp", None) or datetime.now().isoformat()

        # If in position, update trailing stop and check stop loss
        if position == "long":
            if price > highest_price:
                highest_price = price

            # Stop loss triggered
            if stop_loss is not None and price <= entry_price * (1 - stop_loss):
                pnl = price - entry_price
                cumulative_pnl += pnl
                trades.append({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "stop_loss"
                })
                if pnl > 0:
                    win_count += 1
                if log_ml_data:
                    log_row = enrich_features(
                        df, i, symbol, entry_idx, entry_time, entry_price, price, pnl, "stop_loss", signal, hyperparams=hyperparams
                    )
                    log_trade_sample(ml_logfile, log_row, header=ml_feature_fields, verbose=verbose_ml_log)
                position = None
                continue

            # Trailing stop triggered
            if trailing_stop is not None and price <= highest_price * (1 - trailing_stop):
                pnl = price - entry_price
                cumulative_pnl += pnl
                trades.append({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "trailing_stop"
                })
                if pnl > 0:
                    win_count += 1
                if log_ml_data:
                    log_row = enrich_features(
                        df, i, symbol, entry_idx, entry_time, entry_price, price, pnl, "trailing_stop", signal, hyperparams=hyperparams
                    )
                    log_trade_sample(ml_logfile, log_row, header=ml_feature_fields, verbose=verbose_ml_log)
                position = None
                continue

        # Exit signal with ML filter
        if position == "long" and signal == -1:
            should_exit = True
            if ml_filter is not None and hasattr(ml_filter, "should_exit"):
                should_exit = ml_filter.should_exit(df, i, signal)
            if should_exit:
                pnl = price - entry_price
                cumulative_pnl += pnl
                trades.append({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "signal"
                })
                if pnl > 0:
                    win_count += 1
                if log_ml_data:
                    log_row = enrich_features(
                        df, i, symbol, entry_idx, entry_time, entry_price, price, pnl, "signal", signal, hyperparams=hyperparams
                    )
                    log_trade_sample(ml_logfile, log_row, header=ml_feature_fields, verbose=verbose_ml_log)
                position = None

    total_trades = len(trades)
    win_rate = win_count / total_trades if total_trades else 0.0

    # World-class: print summary stats
    print(f"\n[SIMULATOR] {symbol}: {total_trades} trades | PnL: {cumulative_pnl:.4f} | Win rate: {win_rate:.2%}")

    # World-class: autosave all trades for later analysis
    try:
        pd.DataFrame(trades).to_csv(f"trades_{symbol}_debug.csv", index=False)
        print(f"[SIMULATOR] Trade log saved to trades_{symbol}_debug.csv")
    except Exception as e:
        print(f"[SIMULATOR] Could not save trade log for {symbol}: {e}")

    return {
        "trades": trades,
        "pnl": cumulative_pnl,
        "win_rate": win_rate
    }