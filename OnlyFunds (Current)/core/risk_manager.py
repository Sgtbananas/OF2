def calculate_position_size(balance, risk_percent, price):
    """
    Calculates the position size based on account balance, risk percentage, and price.

    Args:
        balance (float): Available account balance (e.g., in USDT).
        risk_percent (float): Risk percentage (0.01 = 1%).
        price (float): Current asset price.

    Returns:
        float: Position size (quantity of asset to buy/sell).
    """
    risk_amount = balance * risk_percent
    position_size = risk_amount / price if price != 0 else 0
    return position_size

def adjust_risk_based_on_profile(config):
    """
    Adjusts the risk parameters in the config dict based on 'risk_profile' key.
    Mutates the config dict in place.

    Args:
        config (dict): The configuration dictionary. Must contain 'risk_profile' key.

    Side effects:
        Updates config dict with keys:
            - "risk_percent"
            - "max_position_size"
            - "stop_loss"
    """
    profile = config.get("risk_profile", "normal").lower()

    if profile == "conservative":
        config["risk_percent"] = 0.01  # 1% risk
        config["max_position_size"] = 0.05
        config["stop_loss"] = 0.02
    elif profile == "aggressive":
        config["risk_percent"] = 0.05  # 5% risk
        config["max_position_size"] = 0.2
        config["stop_loss"] = 0.08
    else:  # normal/default
        config["risk_percent"] = 0.02  # 2% risk
        config["max_position_size"] = 0.1
        config["stop_loss"] = 0.05