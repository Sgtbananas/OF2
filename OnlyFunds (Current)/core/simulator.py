import pandas as pd
from datetime import datetime

def simulate_trades(df, signals, symbol, target, ml_filter=None, stop_loss=None, trailing_stop=None):
    """
    Simulates trades based on provided signals and optional ML filtering and risk management.

    Args:
        df (pd.DataFrame): DataFrame with price data.
        signals (pd.Series): Series with 1 (buy), -1 (sell), 0 (hold) signals.
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        target (float): Target profit/loss per trade.
        ml_filter (optional): ML filter object for filtering trades.
        stop_loss (float, optional): Stop loss as a fraction (e.g. 0.02 for 2%).
        trailing_stop (float, optional): Trailing stop as a fraction (e.g. 0.01 for 1%).

    Returns:
        dict: Contains trades (list of dicts), total PnL, and win rate.
    """
    trades = []
    position = None
    entry_price = 0.0
    highest_price = 0.0  # For trailing stop
    cumulative_pnl = 0.0
    win_count = 0

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
                position = None

    total_trades = len(trades)
    win_rate = win_count / total_trades if total_trades else 0.0

    return {
        "trades": trades,
        "pnl": cumulative_pnl,
        "win_rate": win_rate
    }