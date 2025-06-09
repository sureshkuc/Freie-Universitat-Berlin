"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a configuration class with hyperparameters and constants 
    for use in machine learning or optimization algorithms.
    Logging is set up to capture all log levels to the outputs folder.
Version: 1.0
"""

import os
import logging
from typing import Any


class Config:
    """Configuration class containing model hyperparameters and training constants."""

    TOL: float = 1e-3
    MAX_ITER: int = 100
    LEARNING_RATE: float = 0.1
    EPOCHS: int = 10

    @staticmethod
    def get_all() -> dict[str, Any]:
        """
        Get all configuration parameters.

        Returns:
            dict[str, Any]: Dictionary of all configuration parameters.
        """
        return {
            "TOL": Config.TOL,
            "MAX_ITER": Config.MAX_ITER,
            "LEARNING_RATE": Config.LEARNING_RATE,
            "EPOCHS": Config.EPOCHS,
        }


def setup_logging(log_file: str = "outputs/config.log") -> None:
    """
    Sets up logging to log all levels to the specified file.

    Args:
        log_file (str): The file path where logs will be stored.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.debug("Logging is configured.")


