import yaml
import os
from core.coin_selector import get_top_200_coinex_symbols

def load_config(config_path=None):
    """
    Loads the configuration file (YAML) and fills in required fields.
    Ensures symbols are set, supporting 'symbols_source'.
    """
    # 1. Load YAML
    if config_path is None:
        config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML dict.")

    # 2. Handle symbols population logic
    if not config.get("symbols") and config.get("symbols_source") == "top_200_coinex":
        config["symbols"] = get_top_200_coinex_symbols()
    elif not config.get("symbols") or not isinstance(config["symbols"], list) or not config["symbols"]:
        # As a fallback, pick two major coins
        config["symbols"] = ["BTCUSDT", "ETHUSDT"]

    # 3. Provide sensible defaults for other expected keys
    config.setdefault("all_strategies", ["example_strategy"])
    config.setdefault("strategy", "example_strategy")
    config.setdefault("exchange", "coinex")
    config.setdefault("timeframe", "5m")
    config.setdefault("capital", 1000)
    config.setdefault("risk_profile", "medium")
    config.setdefault("mode", "dry_run")

    # 4. Validate config (basic)
    if not isinstance(config["all_strategies"], list) or not config["all_strategies"]:
        raise ValueError("all_strategies must be a non-empty list.")
    if not isinstance(config["symbols"], list) or not config["symbols"]:
        raise ValueError("symbols must be a non-empty list.")
    if config["mode"] not in ["dry_run", "live", "backtest"]:
        raise ValueError("mode must be one of: dry_run, live, backtest.")

    return config