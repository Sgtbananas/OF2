import ccxt
import requests
from core.data import fetch_klines, add_indicators
from optimizer import optimize_strategies
from core.ml_filter import MLFilter
from core.strategy_loader import get_available_strategies  # <-- Import for automation

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

def get_top_200_coinex_symbols(
    min_usd_volume=50000,
    exclude_stablecoins=True,
    log_prefix="[COIN_SELECTOR]"
):
    """
    Fetch the top 200 CoinEx spot USDT symbols by true market cap using CoinGecko,
    cross-referenced with tradable CoinEx pairs, with robust fallback.
    - Excludes stablecoins by default.
    - Filters by minimum 24h volume (USD) by default.
    - Uses production-ready error handling and logging.

    Args:
        min_usd_volume (float): Minimum 24h volume in USD to include a coin.
        exclude_stablecoins (bool): Whether to filter out stablecoins.
        log_prefix (str): Prefix for log messages.

    Returns: List of symbol strings (format: BTCUSDT, ETHUSDT, ...)
    """
    fallback_symbols = ["BTCUSDT", "ETHUSDT"]

    # List of known stablecoin tickers (add more as needed)
    stablecoins = {
        "USDT", "USDC", "BUSD", "TUSD", "USDP", "USDD", "GUSD", "DAI", "USDS",
        "USDN", "USDX", "EUR", "EURC", "EURS", "EURT", "USDSB", "FDUSD", "PYUSD"
    }

    try:
        # Step 1: Get all CoinEx USDT pairs from CoinEx API
        coinex_url = "https://api.coinex.com/v1/market/list"
        resp = requests.get(coinex_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data or data.get("code") != 0 or "data" not in data:
            print(f"{log_prefix} [ERROR] Unexpected CoinEx API response: {data}. Using fallback symbols.")
            return fallback_symbols
        all_coinex_symbols = [s for s in data["data"] if s.endswith("USDT")]
        coinex_symbol_set = set(all_coinex_symbols)
        print(f"{log_prefix} [DEBUG] CoinEx USDT pairs available: {len(all_coinex_symbols)}")

        # Step 2: Get CoinEx 24h volume data for all pairs (for minimum volume filtering)
        coinex_ticker_url = "https://api.coinex.com/v1/market/ticker/all"
        ticker_resp = requests.get(coinex_ticker_url, timeout=10)
        ticker_resp.raise_for_status()
        ticker_data = ticker_resp.json()
        if (
            not ticker_data
            or ticker_data.get("code") != 0
            or "data" not in ticker_data
            or "ticker" not in ticker_data["data"]
        ):
            print(f"{log_prefix} [ERROR] Unexpected CoinEx ticker API response: {ticker_data}. Using fallback symbols.")
            return fallback_symbols

        ticker_info = ticker_data["data"]["ticker"]
        # DEBUG: Print a sample of the ticker data for inspection
        sample_keys = list(ticker_info.keys())[:5]
        for k in sample_keys:
            print(f"{log_prefix} [TICKER_SAMPLE] {k}: {ticker_info[k]}")

        coinex_volume_map = {}
        for sym, info in ticker_info.items():
            if sym in coinex_symbol_set:
                try:
                    vol = (
                        float(info.get("vol", 0)) or
                        float(info.get("vol24h", 0)) or
                        float(info.get("amount", 0))
                    )
                    last = (
                        float(info.get("last", 0)) or
                        float(info.get("close", 0))
                    )
                    vol_usd = vol * last
                except Exception:
                    vol_usd = 0
                coinex_volume_map[sym] = vol_usd

        print(
            f"{log_prefix} [DEBUG] USDT pairs above min volume {min_usd_volume}: "
            f"{len([s for s in coinex_volume_map if coinex_volume_map[s] >= min_usd_volume])}"
        )

        # Step 3: Fetch top 300 coins by market cap from CoinGecko (to ensure we get 200 tradable on CoinEx)
        cg_url = "https://api.coingecko.com/api/v3/coins/markets"
        cg_params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 300,
            "page": 1,
            "sparkline": "false"
        }
        cg_resp = requests.get(cg_url, params=cg_params, timeout=20)
        cg_resp.raise_for_status()
        cg_data = cg_resp.json()
        if not isinstance(cg_data, list):
            print(f"{log_prefix} [ERROR] Unexpected CoinGecko API response: {cg_data}. Using fallback symbols.")
            return fallback_symbols
        print(f"{log_prefix} [DEBUG] Fetched {len(cg_data)} coins from CoinGecko.")

        top_symbols = []
        seen = set()
        for coin in cg_data:
            cg_symbol = coin.get("symbol", "").upper()
            cg_id = coin.get("id", "")
            cg_name = coin.get("name", "")

            if exclude_stablecoins and (
                cg_symbol in stablecoins or
                cg_id.lower()
                in [
                    "tether",
                    "usd-coin",
                    "binance-usd",
                    "true-usd",
                    "paxos-standard",
                    "dai",
                    "gemini-dollar",
                    "usdd",
                    "usdp",
                    "usds",
                    "usdn",
                    "stasis-eurs",
                    "tether-eurt",
                    "fdusd",
                    "paypal-usd",
                ]
            ):
                continue

            candidate = f"{cg_symbol}USDT"
            if candidate in coinex_symbol_set and candidate not in seen:
                # Check minimum volume on CoinEx
                vol_usd = coinex_volume_map.get(candidate, 0)
                if vol_usd >= min_usd_volume:
                    top_symbols.append(candidate)
                    seen.add(candidate)
                else:
                    print(
                        f"{log_prefix} [DEBUG] Skipping {candidate}: volume ${vol_usd:,.0f} below threshold."
                    )
            if len(top_symbols) >= 200:
                break

        print(f"{log_prefix} [DEBUG] Top symbols selected after all filters: {len(top_symbols)}")
        if len(top_symbols) < 2:
            print(
                f"{log_prefix} [WARN] Unable to find enough top CoinEx symbols by CoinGecko market cap. Using fallback."
            )
            return fallback_symbols
        print(
            f"{log_prefix} [INFO] Returning top {len(top_symbols)} CoinEx symbols by CoinGecko market cap, min ${min_usd_volume:,.0f} 24h vol, stablecoins excluded={exclude_stablecoins}."
        )
        return top_symbols

    except Exception as e:
        print(f"{log_prefix} [ERROR] Exception in get_top_200_coinex_symbols: {e}. Using fallback symbols.")
        return fallback_symbols

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
    min_usd_volume=50000,
    exclude_stablecoins=True,
    config=None,
):
    """
    Select the top N coins based on optimizer results, optionally using ML filtering.
    Always uses CoinGecko + CoinEx cross-referencing for CoinEx.
    Automatically tests all available strategies if none specified.
    """
    # Always use all available strategies unless user explicitly specifies
    strategies_to_test = all_strategies
    if not strategies_to_test or not isinstance(strategies_to_test, list) or len(strategies_to_test) == 0:
        try:
            strategies_to_test = get_available_strategies()
            print(f"[COIN_SELECTOR] [INFO] Automatically testing all available strategies: {strategies_to_test}")
        except Exception as e:
            print(f"[COIN_SELECTOR] [ERROR] Failed to load strategies: {e}")
            strategies_to_test = [strategy_name] if strategy_name else []

    symbols = (
        get_top_200_coinex_symbols(
            min_usd_volume=min_usd_volume,
            exclude_stablecoins=exclude_stablecoins,
        )
        if exchange_name.lower() == "coinex"
        else get_available_symbols(exchange_name, quote)
    )
    ml_filter = MLFilter(model_path=ml_model_path) if ml_enabled else None

    # Always set config['all_strategies'] to strategies_to_test
    if config is None:
        config = {
            "all_strategies": strategies_to_test,
            "timeframe": timeframe,
            "limit": limit,
        }
    else:
        config["all_strategies"] = strategies_to_test
        if timeframe:
            config["timeframe"] = timeframe
        if limit:
            config["limit"] = limit

    results = []
    for symbol in symbols:
        try:
            ccxt_symbol = symbol if '/' in symbol else f"{symbol[:-4]}/{symbol[-4:]}"
            df = fetch_klines(ccxt_symbol, interval=config["timeframe"], limit=config["limit"])
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
            print(f"[COIN_SELECTOR] ⚠️ Skipping {symbol} due to error: {e}")

    # Sort by PnL then win rate
    results.sort(key=lambda x: (x["pnl"], x["win_rate"]), reverse=True)
    return results[:top_n]

if __name__ == "__main__":
    # Example usage: Select top 3 coins for all available strategies with ML filter enabled
    top_coins = select_top_coins(
        top_n=3,
        ml_enabled=True
    )
    for c in top_coins:
        print(f"{c['symbol']}: PnL={c['pnl']:.2f} | WinRate={c['win_rate']:.2%} | Strategies={c['strategies']}")