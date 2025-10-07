import sys
from src.logger import logging

# configure logging (so info messages actually get printed to a log file or console)
logging.basicConfig(
    filename="LOG_FILE_PATH",     # all logs will be written to this file
    level=logging.INFO,       # set logging level
    format='[%(asctime)s] %(levelname)s - %(message)s'  # log message format
)

def error_message_detail(error, error_detail):
    """
    Build a short human-readable error message that includes:
      - the file where the error happened
      - the line number
      - the original error text
    """
    # get exception information (type, value, traceback)
    _, _, exc_tb = error_detail.exc_info()
    # file name where the error happened
    file_name = exc_tb.tb_frame.f_code.co_filename
    # line number
    line_number = exc_tb.tb_lineno
    # formatted message
    error_message = "Error occurred in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        # Build a full formatted message using helper function
        formatted = error_message_detail(error_message, error_detail=error_detail)
        # Initialize base Exception class with this message
        super().__init__(formatted)
        # Save it on the instance too
        self.error_message = formatted

    def __str__(self):
        # Return formatted message when printed
        return self.error_message



