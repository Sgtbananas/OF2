import pandas as pd

def generate_signals(df, config):
    """
    Generate trading signals based on configured strategies.

    Args:
        df (pd.DataFrame): OHLCV data.
        config (dict): Configuration containing strategies.

    Returns:
        pd.Series: Signal series (+1 for buy, -1 for sell, 0 for hold).
    """
    strategies = config.get("strategies", [])
    if not strategies:
        print("⚠️ No strategies provided in config.")
        return pd.Series([0] * len(df), index=df.index)

    # Initialize combined signal to neutral
    combined_signal = pd.Series(0, index=df.index)

    for strategy_name in strategies:
        try:
            strat_module = __import__(f"strategies.{strategy_name}", fromlist=[strategy_name.upper()])
            strategy_class = getattr(strat_module, strategy_name.upper())
            strat_instance = strategy_class()
            strat_signal = strat_instance.generate_signals(df)

            # Combine using simple sum (stacking logic); consider weighted sum if needed
            combined_signal += strat_signal
        except Exception as e:
            print(f"❌ Strategy {strategy_name} failed: {e}")

    # Normalize final signal (could be -2, -1, 0, 1, 2, etc.)
    combined_signal = combined_signal.clip(-1, 1)
    return combined_signal
