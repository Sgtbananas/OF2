import pandas as pd
from datetime import datetime
from core.simulator import simulate_trades

def backtest_strategy(df: pd.DataFrame, signals: pd.Series, symbol: str, target: float):
    """
    Perform a backtest on the provided signals and market data.
    
    Args:
        df (pd.DataFrame): Historical market data.
        signals (pd.Series): Generated trading signals (1 for buy, -1 for sell, 0 for hold).
        symbol (str): The trading pair symbol.
        target (float): Target profit per trade.
    
    Returns:
        dict: Summary of backtest including PnL, win rate, and executed trades.
    """
    if df.empty or signals.empty:
        print(f"❌ No data or signals to backtest for {symbol}")
        return {"symbol": symbol, "pnl": 0.0, "win_rate": 0.0, "trades": []}

    # Use the simulator to calculate the trades based on the signals
    trades = simulate_trades(df, signals, symbol, target)

    if not trades:
        print(f"⚠️ No trades executed for {symbol}.")
        return {"symbol": symbol, "pnl": 0.0, "win_rate": 0.0, "trades": []}

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    win_rate = len(wins) / total_trades if total_trades else 0
    total_pnl = trades_df["pnl"].sum()

    return {
        "symbol": symbol,
        "pnl": total_pnl,
        "win_rate": win_rate,
        "trades": trades
    }

if __name__ == "__main__":
    # Example usage for testing
    import ccxt
    from core.data_handler import fetch_ohlcv
    from core.strategy_loader import load_strategy

    df = fetch_ohlcv("BTC/USDT", "5m", 100)
    strategy = load_strategy("ema")
    signals = strategy.generate_signals(df)
    result = backtest_strategy(df, signals, "BTC/USDT", target=0.02)
    print(result)
