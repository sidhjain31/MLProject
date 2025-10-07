import logging
import os
from datetime import datetime

# 1) decide where to store logs and create that directory
LOG_DIR = os.path.join(os.getcwd(), "logs")    # -> ./logs
os.makedirs(LOG_DIR, exist_ok=True)            # create the folder if it doesn't exist

# 2) build a timestamped filename (use %H for hour, not %h)
LOG_FILE = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)  # full path to the logfile

# 3) configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s:%(lineno)d - %(message)s'
)


