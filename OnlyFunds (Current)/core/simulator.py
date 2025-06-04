import pandas as pd
from datetime import datetime

def simulate_trades(df, signals, symbol, target):
    """
    Simulates trades based on provided signals.

    Args:
        df (pd.DataFrame): DataFrame with price data.
        signals (pd.Series): Series with 1 (buy), -1 (sell), 0 (hold) signals.
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        target (float): Target profit/loss per trade.

    Returns:
        dict: Contains trades (list of dicts), total PnL, and win rate.
    """
    trades = []
    position = None
    entry_price = 0.0
    cumulative_pnl = 0.0
    win_count = 0

    for i in range(len(df)):
        price = df.iloc[i]["close"]
        signal = signals.iloc[i]

        if position is None and signal == 1:  # Enter long
            position = "long"
            entry_price = price

        elif position == "long" and signal == -1:  # Close long
            pnl = price - entry_price
            cumulative_pnl += pnl
            trades.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": price,
                "pnl": pnl,
                "timestamp": datetime.now().isoformat()
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
