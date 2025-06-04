import ccxt
from core.data import fetch_klines, add_indicators
from optimizer import optimize_strategies
from core.ml_filter import MLFilter

def get_available_symbols(exchange_name="coinex", quote="USDT"):
    """Get all active trading pairs for the given quote currency on the specified exchange."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    markets = exchange.load_markets()
    return [s for s in markets if s.endswith(f'/{quote}') and markets[s]['active']]

def select_top_coins(
    strategy_name=None, 
    all_strategies=None, 
    top_n=5, 
    exchange_name="coinex", 
    quote="USDT", 
    timeframe="5m", 
    limit=300, 
    ml_enabled=False, 
    ml_model_path="ml_filter_model.pkl",
    config=None
):
    """
    Select the top N coins based on optimizer results, optionally using ML filtering.
    :param strategy_name: If specified, rank by this strategy only; otherwise all_strategies must be provided.
    :param all_strategies: List of strategies to consider (used if strategy_name is None).
    :param top_n: Number of coins to select.
    :param exchange_name: Exchange to scan.
    :param quote: Quote asset to filter by (e.g. USDT).
    :param timeframe: Candlestick timeframe.
    :param limit: Number of bars to fetch for backtest.
    :param ml_enabled: If True, use MLFilter in optimization.
    :param ml_model_path: Path to ML model for MLFilter.
    :param config: Optional config dict; if provided, overrides function params.
    :return: List of dicts: [{"symbol": ..., "pnl": ..., "win_rate": ..., ...}, ...]
    """
    symbols = get_available_symbols(exchange_name, quote)
    ml_filter = MLFilter(model_path=ml_model_path) if ml_enabled else None

    if config is None:
        config = {
            "all_strategies": all_strategies if all_strategies else [strategy_name],
            "timeframe": timeframe,
            "limit": limit
        }
    else:
        # Override with explicit params if provided
        if all_strategies:
            config["all_strategies"] = all_strategies
        if timeframe:
            config["timeframe"] = timeframe
        if limit:
            config["limit"] = limit

    results = []
    for symbol in symbols:
        try:
            df = fetch_klines(symbol, interval=config["timeframe"], limit=config["limit"])
            df = add_indicators(df)
            if df is None or df.empty:
                continue
            config["symbol"] = symbol
            result_df = optimize_strategies(df, config["all_strategies"], config, ml_filter=ml_filter)
            if not result_df.empty:
                best = result_df.iloc[0]
                results.append({
                    "symbol": symbol,
                    "strategies": best["strategies"],
                    "pnl": best["pnl"],
                    "win_rate": best["win_rate"],
                    "trades": best["trades"]
                })
        except Exception as e:
            print(f"⚠️ Skipping {symbol} due to error: {e}")

    # Sort by PnL then win rate
    results.sort(key=lambda x: (x["pnl"], x["win_rate"]), reverse=True)
    return results[:top_n]

def get_top_200_coinex_symbols():
    """
    Fetch the top 200 CoinEx spot USDT symbols, sorted by market cap if available, falling back to volume.
    Returns: List of symbol strings (format: BTCUSDT, ETHUSDT, ...)
    """
    try:
        exchange = ccxt.coinex()
        markets = exchange.load_markets()
        # Filter to active, spot, USDT quote pairs
        spot_markets = [
            m for m in markets.values()
            if m.get('spot') and m['active'] and m['quote'] == 'USDT'
        ]
        # Try to sort by market cap if available
        def get_market_cap(m):
            market_cap = m.get('info', {}).get('market_cap')
            try:
                return float(market_cap) if market_cap is not None else 0
            except Exception:
                return 0
        has_market_cap = any(get_market_cap(m) > 0 for m in spot_markets)
        if has_market_cap:
            spot_markets.sort(key=lambda x: get_market_cap(x), reverse=True)
        else:
            print("[WARNING] No market cap data available, falling back to sorting by volume.")
            spot_markets.sort(key=lambda x: float(x.get('info', {}).get('volume', 0)), reverse=True)
        top_symbols = [m["symbol"].replace('/', '') for m in spot_markets[:200]]
        if not top_symbols or len(top_symbols) < 2:
            raise Exception("Not enough symbols returned from CoinEx")
        return top_symbols
    except Exception as e:
        print(f"[ERROR] Failed to fetch CoinEx top 200 symbols by market cap: {e}")
        # Fallback to safe default
        return ["BTCUSDT", "ETHUSDT"]

if __name__ == "__main__":
    # Example usage: Select top 3 coins for the "ema" strategy with ML filter enabled
    top_coins = select_top_coins(
        strategy_name="ema",
        top_n=3,
        ml_enabled=True
    )
    for c in top_coins:
        print(f"{c['symbol']}: PnL={c['pnl']:.2f} | WinRate={c['win_rate']:.2%} | Strategies={c['strategies']}")