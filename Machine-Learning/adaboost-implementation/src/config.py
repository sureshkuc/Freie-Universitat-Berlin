"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides configuration settings and logging setup 
    for a machine learning or data processing project. It includes 
    paths, model parameters, and logging initialization.
Version: 1.0
"""

import os
import logging
from typing import Optional


class Config:
    """
    Configuration class that stores constants for paths, 
    model parameters, and logging setup.

    Attributes:
        DATA_PATH (str): Path to the dataset file.
        OUTPUT_DIR (str): Directory where outputs will be saved.
        LOG_FILE (str): Path to the logfile.
        MAX_DEPTH (int): Maximum depth of decision trees.
        N_CLASSIFIERS (int): Number of classifiers to use.
        TEST_SIZE (float): Proportion of dataset to include in test split.
        RANDOM_STATE (int): Seed for random number generator.
    """

    DATA_PATH: str = 'spambase.data'
    OUTPUT_DIR: str = 'outputs'
    LOG_FILE: str = os.path.join(OUTPUT_DIR, 'logfile.log')
    MAX_DEPTH: int = 1
    N_CLASSIFIERS: int = 5
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42


def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration to log messages of all severity levels
    to the specified log file.

    Args:
        log_file (Optional[str]): Path to the log file. Defaults to Config.LOG_FILE.
    """
    try:
        log_file = log_file or Config.LOG_FILE

        # Ensure output directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s — %(levelname)s — %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logging.info("Logging initialized successfully.")

    except Exception as e:
        print(f"Failed to set up logging: {e}")
        raise


# Initialize logging when module is imported
setup_logging()

