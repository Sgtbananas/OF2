import itertools
import pandas as pd
from core.signals import generate_signals
from core.trade import backtest_strategy

def evaluate_pair_strategy(df, pair, strategy_set, config, ml_filter=None):
    """
    Evaluate the PnL and stats for a strategy set on a single trading pair.

    Args:
        df (pd.DataFrame): OHLCV data for the pair.
        pair (str): Trading pair symbol.
        strategy_set (list): List of strategy names.
        config (dict): Config dictionary.
        ml_filter (object, optional): Optional ML filter for trade validation.

    Returns:
        dict: Evaluation result including PnL, trade count, and win rate.
    """
    test_config = config.copy()
    test_config["strategies"] = strategy_set
    signals = generate_signals(df, test_config)
    trades = backtest_strategy(df, signals, test_config, pair, ml_filter)
    pnl = sum(t.get("pnl", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = len(trades) - wins
    win_rate = wins / len(trades) if trades else 0

    return {
        "pair": pair,
        "strategies": strategy_set,
        "pnl": pnl,
        "trades": len(trades),
        "win_rate": win_rate
    }

def get_top_combinations(data_dict, all_strategies, config, top_n=3):
    """
    Evaluate all combinations of strategies across pairs and return top results.

    Args:
        data_dict (dict): Dictionary of pair -> df.
        all_strategies (list): All available strategies.
        config (dict): Config dictionary.
        top_n (int): Number of top combinations to return.

    Returns:
        list: List of top strategy combinations.
    """
    results = []
    strategy_combinations = (
        list(itertools.combinations(all_strategies, 2)) +
        [tuple(all_strategies)]
    )

    for pair, df in data_dict.items():
        for strat_set in strategy_combinations:
            outcome = evaluate_pair_strategy(df, pair, list(strat_set), config)
            results.append(outcome)

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by="pnl", ascending=False).head(top_n)
    return df_result.to_dict(orient="records")
