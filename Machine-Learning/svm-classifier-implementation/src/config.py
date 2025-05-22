"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module sets up configuration parameters and logging utilities 
    for the SVM pipeline project. It ensures that log directories exist, 
    defines constants used across the pipeline, and configures logging 
    to capture all events in a central log file.
Version: 1.0
"""

import os
import logging
from typing import List

# Output directory for logs
LOG_DIR: str = "outputs"
LOG_FILE: str = os.path.join(LOG_DIR, "svm_pipeline.log")

def setup_logging(log_dir: str = LOG_DIR, log_file: str = LOG_FILE) -> None:
    """
    Sets up the logging configuration and ensures the output directory exists.

    Args:
        log_dir (str): The directory where the log file will be stored.
        log_file (str): The complete path of the log file.

    Raises:
        OSError: If the log directory cannot be created.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            filemode="w"
        )
        logging.debug("Logging initialized successfully.")
    except OSError as e:
        print(f"Failed to create log directory '{log_dir}': {e}")
        logging.error("OSError during log directory creation.", exc_info=True)
        raise

# Initialize logging
setup_logging()

# Constants used throughout the pipeline
RANDOM_STATE: int = 1
TEST_SIZE: float = 0.3
C_VALUES: List[float] = [0.1, 0.5, 1, 5, 10, 100]

logging.info("Configuration module loaded successfully.")

