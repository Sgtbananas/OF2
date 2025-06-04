"""
core/visuals.py

This module provides a utility function to print a summary of executed trades.
"""

import pandas as pd

def print_summary(trades):
    """
    Prints a summary of the executed trades.

    Args:
        trades (list or pd.DataFrame): List of trade dictionaries or a DataFrame of trades.
    """
    if trades is None or len(trades) == 0:
        print("❌ No trades executed.")
        return

    # Convert to DataFrame if needed
    if isinstance(trades, list):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades

    if trades_df.empty:
        print("❌ No trades executed.")
        return

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / total_trades if total_trades else 0
    avg_pnl = trades_df["pnl"].mean()

    print("=======================")
    print(f"Final PnL: ${trades_df['pnl'].sum():.2f}")
    print(f"Total Trades: {total_trades} | Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate: {win_rate * 100:.1f}% | Avg Return/Trade: {avg_pnl:.2f} USDT")
    print("=======================")
