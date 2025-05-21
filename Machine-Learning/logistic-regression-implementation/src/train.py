"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions for initializing weights, filtering digits for binary classification,
    relabeling for binary classification, and finding the optimal learning rate for logistic regression.

Version: 1.0
"""

import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.metrics import f1_score

from model import LogisticRegression
from config import CLASS_LABELS


# Configure logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename='outputs/train.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def weight_initialization(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Initialize weights and bias to zeros.

    Args:
        x (np.ndarray): Input feature matrix.

    Returns:
        Tuple[np.ndarray, float]: Initialized weights and bias.
    """
    logging.debug("Initializing weights and bias.")
    return np.zeros((x.shape[1], 1)), 0.0


def filter_digits(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter the dataset to include only two class labels defined in CLASS_LABELS.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered features and labels.
    """
    logging.debug("Filtering dataset to include only class labels: %s", CLASS_LABELS)
    try:
        indices = np.logical_or(y == CLASS_LABELS[0], y == CLASS_LABELS[1])
        return X[indices], y[indices].reshape(-1, 1)
    except Exception as e:
        logging.error("Error while filtering digits: %s", str(e))
        raise


def binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Convert multi-class labels to binary labels: 1 if label > 5, else 0.

    Args:
        y (np.ndarray): Input labels.

    Returns:
        np.ndarray: Binary labels.
    """
    logging.debug("Converting labels to binary.")
    try:
        return np.where(y > 5, 1, 0)
    except Exception as e:
        logging.error("Error during binary label conversion: %s", str(e))
        raise


def find_optimum_lr(
    lr_list: List[float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w: np.ndarray,
    b: float,
    num_iter: int
) -> float:
    """
    Find the optimal learning rate based on F1 score.

    Args:
        lr_list (List[float]): List of learning rates to test.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        w (np.ndarray): Initial weights.
        b (float): Initial bias.
        num_iter (int): Number of iterations for training.

    Returns:
        float: Learning rate that gives the best F1 score.
    """
    logging.debug("Starting search for optimal learning rate.")
    results = []

    try:
        for lr in lr_list:
            logging.info("Testing learning rate: %f", lr)
            model = LogisticRegression(w.copy(), b)
            model.fit(X_train, y_train, lr, num_iter)
            preds = np.where(model.predict(X_test) > 0.5, 1, 0)
            score = f1_score(y_test, preds)
            results.append((score, lr))
            logging.debug("F1 Score for lr=%.5f: %.5f", lr, score)

        results.sort(reverse=True)
        best_lr = results[0][1]
        logging.info("Best learning rate: %.5f with F1 score: %.5f", best_lr, results[0][0])
        return best_lr

    except Exception as e:
        logging.error("Error during learning rate optimization: %s", str(e))
        raise

