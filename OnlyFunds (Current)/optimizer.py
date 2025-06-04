
import itertools
import pandas as pd
from core.signals import generate_signals
from core.trade import backtest_strategy

def optimize_strategies(df, all_strategies, config, ml_filter=None):
    results = []
    for r in range(1, len(all_strategies) + 1):
        for strat_combo in itertools.combinations(all_strategies, r):
            test_config = config.copy()
            test_config["strategies"] = list(strat_combo)
            signal = generate_signals(df, test_config)
            trades = backtest_strategy(df, signal, test_config, config.get("symbol", "TEST"), ml_filter)

            if trades:
                pnl = sum(t.get("pnl", 0) for t in trades if "pnl" in t)
                wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
                losses = sum(1 for t in trades if t.get("pnl", 0) <= 0)
                win_rate = wins / (wins + losses + 1e-9)
                results.append({
                    "strategies": strat_combo,
                    "pnl": pnl,
                    "win_rate": round(win_rate, 2),
                    "trades": len(trades)
                })

    result_df = pd.DataFrame(results).sort_values(by="pnl", ascending=False)
    return result_df
