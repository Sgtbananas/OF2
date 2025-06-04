import yaml
import os
from core.coin_selector import get_top_200_coinex_symbols

def load_config(config_path=None):
    """
    Loads the configuration YAML file and ensures all required fields are present.
    Fills in valid defaults for missing values, and always ensures symbols are set.
    """
    # 1. Load YAML config
    if config_path is None:
        config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML dict.")

    # 2. Symbols logic: prefer config["symbols"], else symbols_source, else fallback
    if not config.get("symbols") or not isinstance(config["symbols"], list) or not config["symbols"]:
        if config.get("symbols_source") == "top_200_coinex":
            config["symbols"] = get_top_200_coinex_symbols()
        else:
            # Fallback: at least BTCUSDT so the app never fails
            config["symbols"] = ["BTCUSDT", "ETHUSDT"]

    # 3. Provide sensible defaults for other expected keys
    config.setdefault("all_strategies", ["conservative", "normal", "aggressive"])
    config.setdefault("strategy", config["all_strategies"][0])
    config.setdefault("exchange", "coinex")
    config.setdefault("timeframe", "5m")
    config.setdefault("capital", 1000)
    config.setdefault("risk_profile", "normal")
    config.setdefault("mode", "dry_run")

    # 4. Validate config
    if not isinstance(config["all_strategies"], list) or not config["all_strategies"]:
        raise ValueError("all_strategies must be a non-empty list.")
    if not isinstance(config["symbols"], list) or not config["symbols"]:
        raise ValueError("symbols must be a non-empty list.")
    if config["mode"] not in ["dry_run", "live", "backtest"]:
        raise ValueError("mode must be one of: dry_run, live, backtest.")
    if config["strategy"] not in config["all_strategies"]:
        config["strategy"] = config["all_strategies"][0]

    return config