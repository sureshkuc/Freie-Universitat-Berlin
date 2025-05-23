"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides utility functions and classes for setting up logging 
    and calculating classification accuracy.
Version: 1.0
"""

import os
import logging
from typing import Union
import numpy as np


class LoggerUtility:
    """A utility class to set up logging with both file and console handlers."""

    def __init__(self, log_file: str = "outputs/app.log") -> None:
        """
        Initialize the logger configuration.

        Args:
            log_file (str): The path to the log file.
        """
        self.log_file = log_file
        self._setup_logging()

    def _setup_logging(self) -> None:
        """
        Set up logging configuration with error handling.
        Logs are written to both console and a specified log file.
        """
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_file),
                    logging.StreamHandler()
                ]
            )
            logging.info("Logging has been configured successfully.")
        except Exception as e:
            print(f"Failed to configure logging: {e}")
            raise


def calculate_accuracy(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calculate the classification accuracy.

    Args:
        y_true (Union[np.ndarray, list]): Ground truth labels.
        y_pred (Union[np.ndarray, list]): Predicted labels.

    Returns:
        float: The accuracy score as a float between 0 and 1.
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        accuracy_score = np.sum(y_true == y_pred) / len(y_true)
        logging.debug(f"Accuracy calculated: {accuracy_score}")
        return accuracy_score
    except Exception as e:
        logging.error(f"Error in calculate_accuracy: {e}", exc_info=True)
        raise

