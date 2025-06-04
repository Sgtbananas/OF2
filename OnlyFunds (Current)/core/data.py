import requests
import pandas as pd
from datetime import datetime

def fetch_klines(symbol: str, interval: str = "1h", limit: int = 100):
    """
    Fetch historical candlestick (kline) data from CoinEx REST API.
    
    Args:
        symbol (str): Trading pair symbol, e.g. "BTCUSDT".
        interval (str): Time interval (e.g., "1m", "5m", "1h").
        limit (int): Number of candles to retrieve.
    
    Returns:
        pd.DataFrame: OHLCV data as a pandas DataFrame.
    """
    url = f"https://api.coinex.com/v1/market/kline?market={symbol}&type={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch data for {symbol}. Status: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json().get("data", [])
    if not data:
        print(f"❌ No data returned for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)

    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe (currently simple examples).
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
    
    Returns:
        pd.DataFrame: DataFrame with additional indicator columns.
    """
    if df.empty:
        return df

    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["rsi_14"] = compute_rsi(df["close"], window=14)
    
    return df

def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a given series.
    
    Args:
        series (pd.Series): Series of closing prices.
        window (int): RSI calculation window.
    
    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

if __name__ == "__main__":
    # Simple test
    df = fetch_klines("BTCUSDT", "5m", 20)
    df = add_indicators(df)
    print(df.tail())
