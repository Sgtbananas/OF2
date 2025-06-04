# core/data_handler.py
import ccxt
import pandas as pd
from datetime import datetime

def fetch_ohlcv(symbol, timeframe="5m", limit=100):
    """
    Fetch historical OHLCV data for a given symbol using ccxt.
    
    Args:
        symbol (str): Trading pair symbol (e.g., "BTC/USDT").
        timeframe (str): Candlestick interval.
        limit (int): Number of data points to retrieve.
    
    Returns:
        pd.DataFrame: DataFrame with timestamp index and OHLCV columns.
    """
    # Initialize exchange with proper rate limit
    exchange = ccxt.coinex({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })
    
    # Load markets to ensure symbol exists
    exchange.load_markets()
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    # Format data into DataFrame
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    return df

if __name__ == "__main__":
    # Quick test for functionality
    df = fetch_ohlcv("BTC/USDT", "1h", 100)
    print(df.tail())
