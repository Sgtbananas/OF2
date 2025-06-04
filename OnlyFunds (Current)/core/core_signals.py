# core/core_signals.py

import pandas as pd

def generate_signals(df, method="ema", **kwargs):
    """
    Generate trading signals based on the specified method.

    Args:
        df (pd.DataFrame): DataFrame containing historical price data with columns:
                           ['Open', 'High', 'Low', 'Close', 'Volume']
        method (str): The method to use for signal generation. Options: 'ema', 'rsi', etc.
        **kwargs: Additional parameters for the chosen method.

    Returns:
        pd.Series: Series of signals (1 = buy, -1 = sell, 0 = hold).
    """
    if method == "ema":
        short = kwargs.get("short", 12)
        long = kwargs.get("long", 26)
        df["ema_short"] = df["Close"].ewm(span=short, adjust=False).mean()
        df["ema_long"] = df["Close"].ewm(span=long, adjust=False).mean()
        signal = (df["ema_short"] > df["ema_long"]).astype(int) - (df["ema_short"] < df["ema_long"]).astype(int)
        return signal

    elif method == "rsi":
        period = kwargs.get("period", 14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        signal = (rsi < 30).astype(int) - (rsi > 70).astype(int)
        return signal

    # Additional strategies (MACD, Bollinger, etc.) can be added here similarly.

    else:
        raise ValueError(f"Unknown method '{method}' for signal generation.")


if __name__ == "__main__":
    # Minimal test to ensure this runs standalone
    data = {
        "Open": [1, 2, 3, 4, 5],
        "High": [1.1, 2.1, 3.1, 4.1, 5.1],
        "Low": [0.9, 1.9, 2.9, 3.9, 4.9],
        "Close": [1, 2, 3, 4, 5],
        "Volume": [100, 110, 120, 130, 140]
    }
    df = pd.DataFrame(data)
    sigs = generate_signals(df, method="ema")
    print(sigs)
