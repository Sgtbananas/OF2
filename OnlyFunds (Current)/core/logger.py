import logging
import os

# Ensure logs directory exists
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "latest.log")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging only once (idempotent for module reloads)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def log_message(msg: str):
    """Log an informational message (both to file and console)."""
    logging.info(msg)

def log_error(msg: str):
    """Log an error message (both to file and console)."""
    logging.error(msg)
