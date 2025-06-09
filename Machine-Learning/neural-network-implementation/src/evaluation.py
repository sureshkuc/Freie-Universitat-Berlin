"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to evaluate machine learning models.
    It includes performance calculation (e.g., accuracy) with logging and exception handling.
Version: 1.0
"""

import os
import logging
import numpy as np
from typing import Any

# Ensure the outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/evaluation.log",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelEvaluator:
    """
    A class to evaluate machine learning models and log performance metrics.
    """

    def __init__(self, model: Any, model_name: str = "Model") -> None:
        """
        Initialize the ModelEvaluator with a model and model name.

        Args:
            model (Any): Trained model with a `.predict()` method.
            model_name (str): Name of the model to use in logs.
        """
        self.model = model
        self.model_name = model_name

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the model using accuracy metric.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): True labels (in range -1 to 1).

        Returns:
            float: Accuracy of the model on the test data.

        Raises:
            Exception: If evaluation fails.
        """
        try:
            logging.debug("Starting evaluation for %s", self.model_name)

            # Convert labels to 0/1 format if originally -1/1
            y_test_binary = ((y_test + 1) // 2).astype(int)
            preds = self.model.predict(X_test)

            accuracy = np.mean(preds == y_test_binary)
            logging.info("%s Accuracy: %.4f", self.model_name, accuracy)

            return accuracy

        except Exception as e:
            logging.exception("Evaluation failed for %s", self.model_name)
            raise

