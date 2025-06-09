"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to train a Multilayer Perceptron (MLP) model.
    It includes utilities for one-hot encoding of labels and training using specified configurations.
Version: 1.0
"""

import os
import logging
import numpy as np
from typing import Any

from multilayerperceptron import MultilayerPerceptron, ActivationFunctions
from config import EPOCHS


# Setup logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/train_mlp.log",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def one_hot_encode(y: np.ndarray) -> np.ndarray:
    """
    Convert integer labels into one-hot encoded format.

    Args:
        y (np.ndarray): Array of integer labels.

    Returns:
        np.ndarray: One-hot encoded 2D array.
    """
    if y.ndim != 1:
        logging.warning("Input label array is not 1-dimensional.")
    y_encoded = np.zeros((y.size, y.max() + 1))
    y_encoded[np.arange(y.size), y] = 1
    return y_encoded


def train_multilayerperceptron_model(X_train: np.ndarray, y_train: np.ndarray) -> MultilayerPerceptron:
    """
    Train a Multilayer Perceptron model on the provided training data.

    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (np.ndarray): Target labels for training.

    Returns:
        MultilayerPerceptron: Trained MLP model.
    """
    try:
        logging.info("Starting preprocessing of training labels.")
        y_train = ((y_train + 1) // 2).astype(int)
        Y_train_encoded = one_hot_encode(y_train)

        logging.info("Initializing MLP model.")
        model = MultilayerPerceptron(
            dimensions=[4, 6, 2],
            activations=[ActivationFunctions.Relu, ActivationFunctions.Softmax]
        )

        logging.info("Beginning training of MLP model.")
        model.train(X_train, Y_train_encoded, epochs=EPOCHS)
        logging.info("Training completed successfully.")
        return model

    except Exception as e:
        logging.exception("Error occurred while training the Multilayer Perceptron model.")
        raise

