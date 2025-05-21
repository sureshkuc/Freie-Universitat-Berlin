"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides configuration settings and logging setup for the application.
    It defines paths for training and testing data, class labels, and initializes a
    logging system to track events and errors.
Version: 1.0
"""

import os
import logging


def setup_logging() -> None:
    """
    Sets up logging to record messages of all severity levels in a log file.

    The log file is created in the 'outputs' directory with the filename 'app.log'.
    If the directory does not exist, it will be created.

    Logs include timestamp, log level, and message.
    """
    try:
        os.makedirs('outputs', exist_ok=True)
        logging.basicConfig(
            filename='outputs/app.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.DEBUG
        )
        logging.info("Logging setup successfully.")
    except Exception as e:
        print(f"Error setting up logging: {e}")
        logging.error("Failed to set up logging.", exc_info=True)


# Constants
TRAIN_PATH: str = './zip.train'
"""Path to the training dataset file."""

TEST_PATH: str = './zip.test'
"""Path to the testing dataset file."""

CLASS_LABELS: list[int] = [1, 9]
"""List of class labels used for classification tasks."""

