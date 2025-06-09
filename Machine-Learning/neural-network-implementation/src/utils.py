"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functionality for preparing a binary classification
    dataset from the Iris dataset, focusing on classifying Setosa vs. non-Setosa.
    The dataset is shuffled and split into training and testing sets.
Version: 1.0
"""

import os
import logging
import numpy as np
from sklearn.datasets import load_iris
from typing import Tuple

# Ensure outputs directory exists for logs
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='outputs/utils.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='a'
)

class SetosaDataPreparer:
    """
    A class to load and prepare the Setosa vs. non-Setosa dataset
    from the Iris dataset for binary classification.
    """

    def __init__(self, test_split_ratio: float = 0.2) -> None:
        """
        Initialize the SetosaDataPreparer.

        Args:
            test_split_ratio (float): Proportion of the dataset to use as test data.
        """
        self.test_split_ratio = test_split_ratio

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the Iris dataset, convert it into a binary classification task
        (Setosa = 1, non-Setosa = -1), shuffle it, and split into training and testing sets.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                X_train, X_test, y_train, y_test
        """
        try:
            logging.info("Loading Iris dataset.")
            iris = load_iris()
            labels = np.copy(iris['target'])

            # Binary classification: Setosa = 1, others = -1
            logging.debug("Transforming labels for binary classification.")
            labels[labels != 0] = -1
            labels[labels == 0] = 1

            logging.debug("Shuffling dataset.")
            idx = np.random.permutation(len(iris.data))
            X_shuffled = iris.data[idx]
            y_shuffled = labels[idx]

            split_index = int((1 - self.test_split_ratio) * len(X_shuffled))
            X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
            y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]

            logging.info("Data preparation completed successfully.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("An error occurred during data preparation.", exc_info=True)
            raise e

