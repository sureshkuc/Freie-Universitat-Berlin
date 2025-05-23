"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to load data,
    train an AdaBoost model, and evaluate its performance.
Version: 1.0
"""

import os
import logging
import traceback
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .adaboost import Adaboost
from .utils import accuracy
from .config import DATA_PATH, TEST_SIZE, RANDOM_STATE


# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='outputs/train.log',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Trainer:
    """
    Trainer class to handle data loading, model training, and prediction.
    """

    def __init__(self) -> None:
        """
        Initializes a Trainer instance.
        """
        self.model = Adaboost()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads and splits the dataset into training and testing sets.

        Returns:
            Tuple containing X_train, X_test, y_train, y_test arrays.

        Raises:
            Exception: If there is an error reading or processing the data.
        """
        try:
            data = np.array(pd.read_csv(DATA_PATH, header=None))
            X = data[:, :-1]
            y = np.where(data[:, -1] == 1, 1, -1)  # Convert labels to -1 and +1
            logging.info("Data loaded successfully.")
            return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        except Exception as e:
            logging.error("Error loading data: %s", e)
            logging.debug("Traceback:\n%s", traceback.format_exc())
            raise

    def train_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trains the Adaboost model and makes predictions on the test set.

        Returns:
            A tuple of true labels and predicted labels (y_test, y_pred).

        Raises:
            Exception: If training or prediction fails.
        """
        try:
            X_train, X_test, y_train, y_test = self.load_data()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            logging.info("Model training and prediction completed.")
            return y_test, y_pred
        except Exception as e:
            logging.error("Error during model training: %s", e)
            logging.debug("Traceback:\n%s", traceback.format_exc())
            raise


# Example usage (if used as a script)

try:
	trainer = Trainer()
	y_test, y_pred = trainer.train_model()
	acc = accuracy(y_test, y_pred)
	logging.info(f"Model Accuracy: {acc:.4f}")
except Exception as e:
	logging.critical("Critical error during training process: %s", e)
	logging.debug("Traceback:\n%s", traceback.format_exc())

