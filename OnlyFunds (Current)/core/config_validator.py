def validate_config(config):
    required_keys = ["mode", "risk", "target", "all_strategies", "symbols"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if not isinstance(config["all_strategies"], list) or not config["all_strategies"]:
        raise ValueError("all_strategies must be a non-empty list.")
    if not isinstance(config["symbols"], list) or not config["symbols"]:
        raise ValueError("symbols must be a non-empty list.")
    # Further checks
    if config["mode"] not in ["dry_run", "live", "backtest"]:
        raise ValueError("mode must be one of: dry_run, live, backtest.")
    if config["risk"] not in ["normal", "high", "low"]:
        raise ValueError("risk must be one of: normal, high, low.")
    if not (0 < float(config["target"]) < 1):
        raise ValueError("target must be a float between 0 and 1.")
    return True