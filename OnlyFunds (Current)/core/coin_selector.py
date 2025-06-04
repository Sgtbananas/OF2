import ccxt

def get_top_200_coinex_symbols():
    """
    Fetches the top 200 trading pairs on CoinEx by 24h volume, as a proxy for market cap.
    Returns: List of symbols in 'BTC/USDT' format.
    """
    exchange = ccxt.coinex()
    markets = exchange.load_markets()
    # Filter for spot markets only
    spot_markets = [m for m in markets.values() if m.get('spot') and m['active']]
    # Sort by 24h volume (descending)
    spot_markets.sort(key=lambda x: x.get('info', {}).get('volume', 0), reverse=True)
    top_200 = spot_markets[:200]
    return [m['symbol'] for m in top_200]