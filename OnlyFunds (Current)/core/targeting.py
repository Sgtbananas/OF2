def get_daily_profit_target(equity, mode):
    if mode == "conservative":
        return equity * 0.05  # 5% of equity per day
    elif mode == "normal":
        return equity * 0.10  # 10% of equity per day
    elif mode == "aggressive":
        return equity * 20.0  # 20x account balance per day
    return equity * 0.05  # fallback