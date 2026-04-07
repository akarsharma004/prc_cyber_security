import logging
import os
import datetime

# Create log file name
LOG_FILE = f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Correct directory path
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Final log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)