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
    # Determine the risk amount in dollars
    risk_amount = balance * risk_percent

    # Calculate position size
    position_size = risk_amount / price if price != 0 else 0

    return position_size


def adjust_risk_based_on_profile(risk_profile):
    """
    Adjusts the risk percentage based on the selected risk profile.

    Args:
        risk_profile (str): One of "conservative", "normal", or "aggressive".

    Returns:
        float: Risk percentage (e.g., 0.01 for 1% risk).
    """
    risk_profile = risk_profile.lower()

    if risk_profile == "conservative":
        return 0.01  # 1% risk
    elif risk_profile == "normal":
        return 0.02  # 2% risk
    elif risk_profile == "aggressive":
        return 0.05  # 5% risk
    else:
        return 0.02  # Default to 2% risk if invalid profile

