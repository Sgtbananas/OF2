import os
import time

def is_file_fresh(path, max_age_hours=24):
    """
    Checks whether a file exists and is not older than max_age_hours.
    """
    if not os.path.exists(path):
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds < max_age_hours * 3600

def ensure_file(path, generator_func, *args, **kwargs):
    """
    If the file at `path` is missing or stale, run `generator_func` to regenerate it.
    """
    if not is_file_fresh(path, kwargs.get("max_age_hours", 24)):
        print(f"[UTILS] File {path} is missing or stale. Regenerating...")
        generator_func(*args)
    else:
        print(f"[UTILS] Using fresh file: {path}")
