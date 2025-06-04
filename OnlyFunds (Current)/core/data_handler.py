import ccxt
import pandas as pd

def fetch_ohlcv(symbol, timeframe="5m", limit=300, exchange=None):
    """
    Fetch historical OHLCV data for a given symbol using ccxt.
    Returns: pd.DataFrame with timestamp index and OHLCV columns.
    """
    # Accepts optional ccxt exchange instance (for reuse)
    if exchange is None:
        exchange = ccxt.coinex({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
    # Ensure market is loaded
    if not hasattr(exchange, "markets") or not exchange.markets:
        exchange.load_markets()
    # ccxt expects 'BTC/USDT' format
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def fetch_all_symbols_by_volume(exchange=None, top_n=200):
    """
    Fetches all active CoinEx spot symbols, sorted by 24h volume (descending).
    Returns: List of symbol strings.
    """
    if exchange is None:
        exchange = ccxt.coinex()
    markets = exchange.load_markets()
    spot_markets = [m for m in markets.values() if m.get('spot') and m['active']]
    spot_markets.sort(key=lambda x: float(x.get('info', {}).get('volume', 0)), reverse=True)
    return [m["symbol"] for m in spot_markets[:top_n]]