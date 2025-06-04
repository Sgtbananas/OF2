def validate_config(config):
    required_keys = ["mode", "risk", "target", "all_strategies", "symbols"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if not isinstance(config["all_strategies"], list) or not config["all_strategies"]:
        raise ValueError("all_strategies must be a non-empty list.")
    if not isinstance(config["symbols"], list) or not config["symbols"]:
        raise ValueError("symbols must be a non-empty list.")
    # You can add further checks as you see fit
    return True