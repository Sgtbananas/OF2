def format_symbol(symbol):
    """
    Ensures symbol is in standard format. Example: 'btc/usdt' → 'BTCUSDT'
    """
    return symbol.replace("/", "").upper()
