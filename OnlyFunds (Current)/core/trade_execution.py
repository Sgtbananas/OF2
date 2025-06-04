import ccxt
from datetime import datetime
from core.logger import log_message
from core.config import load_config

def place_live_order(symbol, side, amount, price=None, order_type="market"):
    """
    Places a live order via CoinEx API using ccxt.

    Args:
        symbol (str): e.g., "BTC/USDT".
        side (str): "buy" or "sell".
        amount (float): Amount to buy or sell.
        price (float, optional): For limit orders.
        order_type (str): "market" or "limit".

    Returns:
        dict: Order execution result.
    """
    config = load_config("config.yaml")
    api_key = config["coinex"]["api_key"]
    api_secret = config["coinex"]["api_secret"]

    exchange = ccxt.coinex({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })

    log_message(f"🚀 [LIVE ORDER] {side.upper()} {amount} {symbol} at {price if price else 'market price'}")

    try:
        params = {}
        if order_type == "market":
            order = exchange.create_market_order(symbol, side, amount, params)
        elif order_type == "limit":
            if not price:
                raise ValueError("Limit orders require a price.")
            order = exchange.create_limit_order(symbol, side, amount, price, params)
        else:
            raise ValueError("Invalid order_type. Use 'market' or 'limit'.")

        log_message(f"✅ Order executed: {order}")
        return order
    except Exception as e:
        log_message(f"❌ Order execution error: {e}")
        return {"error": str(e)}


def simulate_order(symbol, side, amount, price=None):
    """
    Simulates an order for dry run/backtest modes.

    Args:
        symbol (str): e.g., "BTC/USDT".
        side (str): "buy" or "sell".
        amount (float): Amount to buy or sell.
        price (float, optional): Simulated price.

    Returns:
        dict: Simulated order data.
    """
    log_message(f"💡 [SIMULATED ORDER] {side.upper()} {amount} {symbol} at {price if price else 'market price'}")

    order_data = {
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "price": price if price else "market price",
        "status": "simulated",
        "timestamp": datetime.now().isoformat()
    }
    return order_data
