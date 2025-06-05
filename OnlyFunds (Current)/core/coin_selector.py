import ccxt
import requests
from core.data import fetch_klines, add_indicators
from optimizer import optimize_strategies
from core.ml_filter import MLFilter

def get_available_symbols(exchange_name="coinex", quote="USDT"):
    """Get all active trading pairs for the given quote currency on the specified exchange."""
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
        markets = exchange.load_markets()
        symbols = [s for s in markets if s.endswith(f'/{quote}') and markets[s]['active']]
        if not symbols:
            print(f"[ERROR] No active {quote} pairs found on {exchange_name}. Using fallback.")
            return ["BTC/USDT", "ETH/USDT"]
        return symbols
    except Exception as e:
        print(f"[ERROR] Could not fetch symbols from {exchange_name}: {e}. Using fallback.")
        return ["BTC/USDT", "ETH/USDT"]

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
    Fetch the top 200 CoinEx spot USDT symbols directly from CoinEx API,
    sorted by 24h volume, with robust fallback.

    Returns: List of symbol strings (format: BTCUSDT, ETHUSDT, ...)
    """
    url = "https://api.coinex.com/v1/market/list"
    fallback_symbols = ["BTCUSDT", "ETHUSDT"]

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data or data.get("code") != 0 or "data" not in data:
            print(f"[ERROR] Unexpected API response: {data}. Using fallback symbols.")
            return fallback_symbols
        markets = data["data"]
        # Filter to active, spot, USDT quote pairs
        spot_usdt = [
            m for m in markets
            if m.get("status") == "normal" and m.get("quote_asset", "").upper() == "USDT"
        ]
        print(f"[DEBUG] Fetched {len(spot_usdt)} active USDT spot pairs from CoinEx API.")
        if not spot_usdt:
            print("[ERROR] CoinEx API returned no spot pairs. Using fallback symbols.")
            return fallback_symbols

        # Sort by 24h volume (descending) if available
        def parse_volume(m):
            try:
                return float(m.get("volume_24h", 0))
            except Exception:
                return 0.0

        spot_usdt.sort(key=parse_volume, reverse=True)
        top_symbols = [m["market"].replace('/', '') for m in spot_usdt[:200]]

        # Final fallback if result is empty or too short
        if not top_symbols or len(top_symbols) < 2:
            print(f"[ERROR] Not enough symbols returned from CoinEx API: {len(top_symbols)}. Using fallback symbols.")
            return fallback_symbols

        print(f"[DEBUG] Returning top {len(top_symbols)} CoinEx symbols.")
        return top_symbols
    except Exception as e:
        print(f"[ERROR] Exception fetching CoinEx symbols: {e}. Using fallback symbols.")
        return fallback_symbols

if __name__ == "__main__":
    # Example usage: Select top 3 coins for the "ema" strategy with ML filter enabled
    top_coins = select_top_coins(
        strategy_name="ema",
        top_n=3,
        ml_enabled=True
    )
    for c in top_coins:
        print(f"{c['symbol']}: PnL={c['pnl']:.2f} | WinRate={c['win_rate']:.2%} | Strategies={c['strategies']}")