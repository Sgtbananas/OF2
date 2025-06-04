import yaml
import os
from core.config_validator import validate_config
from core.coin_selector import get_top_200_coinex_symbols

DEFAULT_CONFIG = {
    "mode": "dry_run",
    "risk": "normal",
    "target": 0.01,
    "symbols_source": "top_200_coinex",
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "all_strategies": ["ema", "rsi", "macd", "bollinger", "trend_score"],
    "ml_filter": True,
    "ml_threshold": 0.6,
    "timeframe": "5min",
    "limit": 300,
    "initial_capital": 10,
    "threshold": 0.5,
    "live": False,
    "coinex": {
        "api_key": "YOUR_API_KEY",
        "api_secret": "YOUR_SECRET"
    },
    "trailing_stop_pct": 0.03,
    "trailing_trigger_pct": 0.02,
}

def load_config(path="OnlyFunds (Current)/config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    # Apply defaults for any missing values
    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
    # Nested defaults
    if "coinex" not in config:
        config["coinex"] = DEFAULT_CONFIG["coinex"]
    else:
        for ck, cv in DEFAULT_CONFIG["coinex"].items():
            if ck not in config["coinex"]:
                config["coinex"][ck] = cv
    # Dynamic symbol population
    if config.get("symbols_source") == "top_200_coinex":
        config["symbols"] = get_top_200_coinex_symbols()
    # Validate config
    validate_config(config)
    return config