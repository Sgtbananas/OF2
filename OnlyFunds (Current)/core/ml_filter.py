import importlib
import os

STRATEGY_DIR = "strategies"

def load_strategy(name):
    """
    Dynamically load a strategy class by name.
    Args:
        name (str): Lowercase version of the filename in the strategies dir.
    Returns:
        object: Instantiated strategy class.
    Raises:
        ImportError if not found.
    """
    try:
        module_path = f"{STRATEGY_DIR}.{name}"
        strategy_module = importlib.import_module(module_path)
        # Class name convention: uppercase version of filename
        class_name = name.upper()
        strategy_class = getattr(strategy_module, class_name)
        return strategy_class()
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Strategy '{name}' could not be loaded: {e}")

def get_available_strategies():
    """
    List all available strategy modules in the strategies directory.
    Returns: list of strategy names (no .py extension).
    """
    strategy_files = [
        f[:-3] for f in os.listdir(STRATEGY_DIR)
        if f.endswith(".py") and f != "__init__.py"
    ]
    return strategy_files

def strategy_doc(name):
    """
    Returns the docstring of a loaded strategy's class, or None if not found.
    """
    try:
        strat = load_strategy(name)
        return strat.__doc__
    except Exception:
        return None

# PATCH: No changes needed, but this file is kept for completeness and future extensibility.