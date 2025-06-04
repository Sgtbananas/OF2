import os
from datetime import datetime

LOG_DIR = "logs"

def log_message(message: str, log_file: str = None):
    """
    Logs a message to a log file with a timestamp.
    
    Args:
        message (str): The message to log.
        log_file (str, optional): Filename of the log file. Defaults to today's date.
    """
    # Ensure the log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Default to daily log file if not specified
    if not log_file:
        log_file = datetime.now().strftime("%Y-%m-%d") + ".log"
    
    log_path = os.path.join(LOG_DIR, log_file)
    
    # Format the log message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}\n"
    
    # Write to file
    with open(log_path, "a") as f:
        f.write(formatted_message)
    
    # Also print to console for visibility
    print(formatted_message.strip())
