"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to train a Linear SVM model using soft-margin formulation.
    It includes error handling and logs all events during training for debugging and audit purposes.
Version: 1.0
"""

import logging
import os
from typing import Tuple
from model import LinearSVM
import numpy as np

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join("outputs", "train.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(X_train: np.ndarray, y_train: np.ndarray, C: float) -> Tuple[np.ndarray, float]:
    """
    Trains a soft-margin Linear SVM using the provided training data and regularization parameter.

    Args:
        X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        y_train (np.ndarray): Training labels of shape (n_samples,).
        C (float): Regularization parameter controlling the trade-off between margin size and misclassification.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the learned weight vector and bias term.

    Raises:
        Exception: If the training process encounters an error.
    """
    logger.info("Starting model training...")
    try:
        w, b = LinearSVM.train_soft(X_train, y_train, C)
        logger.info("Model training completed successfully.")
        return w, b
    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        raise

