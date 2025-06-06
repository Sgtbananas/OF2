import pandas as pd
from datetime import datetime
import csv
import os

def log_trade_sample(logfile, row, header=None):
    """Append a trade sample (features + label) to the given CSV file."""
    try:
        file_exists = os.path.isfile(logfile)
        with open(logfile, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header or row.keys())
            if not file_exists or os.stat(logfile).st_size == 0:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[ML_LOGGER] Error logging trade sample: {e}")

def simulate_trades(
    df, signals, symbol, target, ml_filter=None,
    stop_loss=None, trailing_stop=None,
    log_ml_data: bool = False,
    ml_logfile: str = "ml_training_data.csv"
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

    Returns:
        dict: Contains trades (list of dicts), total PnL, and win rate.
    """
    trades = []
    position = None
    entry_price = 0.0
    highest_price = 0.0  # For trailing stop
    cumulative_pnl = 0.0
    win_count = 0

    # For ML logging: collect all possible feature columns
    ml_feature_fields = [
        "symbol", "timestamp", "entry_idx", "exit_idx",
        "entry_price", "exit_price", "pnl", "reason",
        "signal", "close", "high", "low", "volume"
        # Add more indicators as needed below
    ]
    # Add all indicator columns from df if present (excluding duplicates)
    ml_feature_fields += [col for col in df.columns if col not in ml_feature_fields]

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
                    log_row = {
                        "symbol": symbol,
                        "timestamp": entry_time,
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                        "reason": "stop_loss",
                        "signal": signal,
                        "close": price,
                        "high": df.iloc[i].get("high", None),
                        "low": df.iloc[i].get("low", None),
                        "volume": df.iloc[i].get("volume", None),
                        # Add indicators here as needed
                    }
                    # Add additional features
                    for feat in df.columns:
                        if feat not in log_row:
                            log_row[feat] = df.iloc[i].get(feat, None)
                    log_row["label"] = 1 if pnl > 0 else 0
                    log_trade_sample(ml_logfile, log_row, header=ml_feature_fields + ["label"])
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
                    log_row = {
                        "symbol": symbol,
                        "timestamp": entry_time,
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                        "reason": "trailing_stop",
                        "signal": signal,
                        "close": price,
                        "high": df.iloc[i].get("high", None),
                        "low": df.iloc[i].get("low", None),
                        "volume": df.iloc[i].get("volume", None),
                        # Add indicators here as needed
                    }
                    for feat in df.columns:
                        if feat not in log_row:
                            log_row[feat] = df.iloc[i].get(feat, None)
                    log_row["label"] = 1 if pnl > 0 else 0
                    log_trade_sample(ml_logfile, log_row, header=ml_feature_fields + ["label"])
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
                    log_row = {
                        "symbol": symbol,
                        "timestamp": entry_time,
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                        "reason": "signal",
                        "signal": signal,
                        "close": price,
                        "high": df.iloc[i].get("high", None),
                        "low": df.iloc[i].get("low", None),
                        "volume": df.iloc[i].get("volume", None),
                        # Add indicators here as needed
                    }
                    for feat in df.columns:
                        if feat not in log_row:
                            log_row[feat] = df.iloc[i].get(feat, None)
                    log_row["label"] = 1 if pnl > 0 else 0
                    log_trade_sample(ml_logfile, log_row, header=ml_feature_fields + ["label"])
                position = None

    total_trades = len(trades)
    win_rate = win_count / total_trades if total_trades else 0.0

    return {
        "trades": trades,
        "pnl": cumulative_pnl,
        "win_rate": win_rate
    }