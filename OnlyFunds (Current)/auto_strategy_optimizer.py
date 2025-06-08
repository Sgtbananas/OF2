import pandas as pd
from core.data import fetch_klines, add_indicators
from optimizer import optimize_strategies
from core.ml_filter import MLFilter  # PATCH: allow MLFilter use in optimizer

def run_auto_optimization(symbols, config):
    all_strategies = config["all_strategies"]
    timeframe = config["timeframe"]
    limit = config["limit"]
    results = []

    # PATCH: Optionally load MLFilter for optimizer if enabled/configured
    ml_filter = None
    ml_cfg = config.get("ml_filter", {})
    ml_enabled = ml_cfg.get("enabled", False) if isinstance(ml_cfg, dict) else bool(ml_cfg)
    model_path = ml_cfg.get("model_path", "ml_filter_model.pkl") if isinstance(ml_cfg, dict) else "ml_filter_model.pkl"
    if ml_enabled:
        try:
            ml_filter = MLFilter(model_path=model_path)
            print(f"✅ ML filter loaded for optimizer from {model_path}")
        except Exception as e:
            print(f"❌ MLFilter failed for optimizer: {e}")
            ml_filter = None

    for symbol in symbols:
        print(f"🔍 Optimizing strategies for {symbol}...")
        df = fetch_klines(symbol, interval=timeframe, limit=limit)
        df = add_indicators(df)
        if df is None or df.empty:
            print(f"⚠️ Skipping {symbol} — no data.")
            continue

        config["symbol"] = symbol
        # PATCH: Always pass ml_filter to optimizer
        result_df = optimize_strategies(df, all_strategies, config, ml_filter=ml_filter)
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