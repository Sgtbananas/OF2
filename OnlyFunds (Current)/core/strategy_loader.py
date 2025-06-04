import importlib
import os

STRATEGY_DIR = "strategies"

def load_strategy(name):
    """
    Dynamically load a strategy class by name.

    Args:
        name (str): Strategy name (lowercase, matching the .py file).

    Returns:
        object: Instantiated strategy class.
    """
    try:
        module_path = f"{STRATEGY_DIR}.{name}"
        strategy_module = importlib.import_module(module_path)

        # Class name is uppercase version of the filename
        class_name = name.upper()
        strategy_class = getattr(strategy_module, class_name)

        return strategy_class()
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Strategy '{name}' could not be loaded: {e}")

def get_available_strategies():
    """
    List available strategy modules in the strategies directory.

    Returns:
        list: Available strategy names (no .py extensions).
    """
    strategy_files = [
        f[:-3] for f in os.listdir(STRATEGY_DIR)
        if f.endswith(".py") and f != "__init__.py"
    ]
    return strategy_files
