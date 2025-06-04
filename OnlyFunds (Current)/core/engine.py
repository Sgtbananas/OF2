import pandas as pd
from core.data_handler import fetch_ohlcv
from core.simulator import simulate_trades
from core.strategy_loader import load_strategy
from core.order_manager import execute_trade, dry_run_trade, log_backtest
from core.logger import log_message
from core.visuals import print_summary
from utils import format_symbol


def run_bot(config):
    """
    Main bot execution function.
    """
    mode = config["mode"]
    risk = config["risk"]
    target = float(config["target"])
    symbols = config["symbols"]

    ledger = []

    for symbol in symbols:
        log_message(f"Fetching data for {symbol}...")
        df = fetch_ohlcv(symbol)

        if df is None or df.empty:
            log_message(f"❌ Failed to fetch data for {symbol}")
            continue

        best_result = None
        strategies = {}

        for strat_name in config["all_strategies"]:
            try:
                strategy = load_strategy(strat_name)
                signals = strategy.generate_signals(df)
                result = simulate_trades(df, signals, symbol, target)
                strategies[strat_name] = result
            except Exception as e:
                log_message(f"⚠️ Strategy {strat_name} failed: {e}")

        if not strategies:
            log_message(f"❌ No successful strategies for {symbol}. Skipping.")
            continue

        # Select best strategy by PnL
        best_strategy = max(strategies.items(), key=lambda x: x[1]["pnl"])[0]
        best_result = strategies[best_strategy]

        log_message(
            f"✅ {symbol}: Best strategy → {best_strategy} | "
            f"PnL: {best_result['pnl']:.2f} | Win rate: {best_result['win_rate']:.2f}"
        )

        # Save trade results based on mode
        if mode == "live":
            execute_trade(symbol, best_result.get("trades", []))
        elif mode == "dry_run":
            dry_run_trade(symbol, best_result.get("trades", []))
        elif mode == "backtest":
            log_backtest(symbol, best_result.get("trades", []))

        # Add to ledger
        ledger.extend(best_result["trades"])

    # Final summary
    print_summary(pd.DataFrame(ledger))
