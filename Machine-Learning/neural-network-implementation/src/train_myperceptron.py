"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a class for training a custom MyPerceptron model 
    using the given training data. It includes error handling and logs
    all activity into the outputs directory.
Version: 1.0
"""

import os
import logging
from typing import Any
from myperceptron import MyPerceptron
import numpy as np

# Ensure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/train_myperceptron.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)
logger = logging.getLogger(__name__)


class MyPerceptronTrainer:
    """
    Trainer class for MyPerceptron model.
    """

    def __init__(self) -> None:
        """
        Initialize the MyPerceptron model.
        """
        self.model = MyPerceptron()
        logger.info("Initialized MyPerceptron model.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> MyPerceptron:
        """
        Train the MyPerceptron model using training data.

        Args:
            X_train (np.ndarray): Feature matrix for training.
            y_train (np.ndarray): Target vector for training.

        Returns:
            MyPerceptron: Trained model instance.

        Raises:
            Exception: If training fails.
        """
        try:
            logger.info("Starting training MyPerceptron model.")
            self.model.fit(X_train, y_train)
            logger.info("Training completed successfully.")
            return self.model
        except Exception as e:
            logger.exception("Error occurred while training MyPerceptron model.")
            raise e


def train_myperceptron_model(X_train: np.ndarray, y_train: np.ndarray) -> MyPerceptron:
    """
    Convenience function to train MyPerceptron using the trainer class.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        MyPerceptron: Trained model.
    """
    trainer = MyPerceptronTrainer()
    return trainer.train(X_train, y_train)

