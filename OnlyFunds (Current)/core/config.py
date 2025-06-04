import yaml

def load_config(file_path: str) -> dict:
    """
    Load a YAML configuration file.
    
    Args:
        file_path (str): Path to the YAML file.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: dict, file_path: str):
    """
    Save a dictionary to a YAML configuration file.
    
    Args:
        config (dict): Configuration dictionary.
        file_path (str): Path to the YAML file.
    """
    with open(file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    # Example usage
    config_data = load_config("config.yaml")
    print(config_data)
