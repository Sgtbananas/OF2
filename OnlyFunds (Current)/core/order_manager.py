import json
from datetime import datetime

ORDERS_FILE = "orders_log.json"


def load_orders():
    try:
        with open(ORDERS_FILE, "r") as f:
            orders = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        orders = []
    return orders


def save_orders(orders):
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=4)


def execute_trade(symbol, trades):
    """
    Records executed trades in live mode.
    """
    orders = load_orders()
    for trade in trades:
        trade["symbol"] = symbol
        trade["type"] = "live"
        trade["timestamp"] = datetime.now().isoformat()
        orders.append(trade)
    save_orders(orders)


def dry_run_trade(symbol, trades):
    """
    Records executed trades in dry run mode.
    """
    orders = load_orders()
    for trade in trades:
        trade["symbol"] = symbol
        trade["type"] = "dry_run"
        trade["timestamp"] = datetime.now().isoformat()
        orders.append(trade)
    save_orders(orders)


def log_backtest(symbol, trades):
    """
    Records backtested trades.
    """
    orders = load_orders()
    for trade in trades:
        trade["symbol"] = symbol
        trade["type"] = "backtest"
        trade["timestamp"] = datetime.now().isoformat()
        orders.append(trade)
    save_orders(orders)
