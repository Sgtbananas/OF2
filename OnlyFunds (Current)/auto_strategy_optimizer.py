
import pandas as pd
from core.data import fetch_klines, add_indicators
from optimizer import optimize_strategies

def run_auto_optimization(symbols, config):
    all_strategies = config["all_strategies"]
    timeframe = config["timeframe"]
    limit = config["limit"]
    results = []

    for symbol in symbols:
        print(f"🔍 Optimizing strategies for {symbol}...")
        df = fetch_klines(symbol, interval=timeframe, limit=limit)
        df = add_indicators(df)
        if df is None or df.empty:
            print(f"⚠️ Skipping {symbol} — no data.")
            continue

        config["symbol"] = symbol
        result_df = optimize_strategies(df, all_strategies, config)
        if not result_df.empty:
            best = result_df.iloc[0]
            results.append({
                "pair": symbol,
                "strategies": best["strategies"],
                "pnl": best["pnl"],
                "win_rate": best["win_rate"],
                "trades": best["trades"]
            })
            print(f"✅ {symbol}: Best stack → {best['strategies']} | PnL: {best['pnl']:.2f} | Win rate: {best['win_rate']}")
        else:
            print(f"❌ No results for {symbol}")

    return pd.DataFrame(results)
