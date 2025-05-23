"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a configuration class and logging setup 
    for a machine learning project using decision trees. It includes
    data paths, model hyperparameters, and logging configurations.
Version: 1.0
"""

import os
import logging
from typing import Optional


class Config:
    """
    Configuration class for setting paths, model parameters, and logging.

    Attributes:
        DATA_PATH (str): Path to the dataset.
        LOG_DIR (str): Directory to store log files.
        LOG_FILE (str): Full path to the log file.
        MAX_DEPTH (int): Maximum depth of the decision tree.
        MIN_SAMPLES_LEAF (int): Minimum samples per leaf node.
        RANDOM_STATE (int): Random seed for reproducibility.
    """
    DATA_PATH: str = 'spambase.data'
    LOG_DIR: str = 'outputs'
    LOG_FILE: str = os.path.join(LOG_DIR, 'logfile.log')

    MAX_DEPTH: int = 10
    MIN_SAMPLES_LEAF: int = 1
    RANDOM_STATE: int = 0


def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Sets up logging to file and console with all error levels.

    Args:
        log_file (Optional[str]): Path to the log file. If None, uses default from Config.
    """
    log_file = log_file or Config.LOG_FILE

    # Ensure log directory exists
    try:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating log directory '{Config.LOG_DIR}': {e}")
        raise

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.debug("Logging is set up.")
    logging.info("Log file: %s", log_file)


# Setup logging when module is imported
try:
    setup_logging()
except Exception as e:
    print(f"Failed to setup logging: {e}")

