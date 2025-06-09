"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a custom Perceptron class (`MyPerceptron`) to train a simple binary classifier using
    an iterative learning approach. It includes methods for fitting the model, predicting new samples,
    and evaluating accuracy. The module handles logging and error reporting for all major steps.
Version: 1.0
"""

import numpy as np
import logging
import os
from typing import Optional

# Constants
TOL: float = 1e-3
MAX_ITER: int = 1000

# Setup logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/perceptron.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

class MyPerceptron:
    """
    A simple implementation of the Perceptron binary classifier.

    Attributes:
        tol (float): Tolerance threshold to stop iterations early.
        max_iter (int): Maximum number of iterations for training.
        sep_hp (Optional[np.ndarray]): The learned separating hyperplane.
    """

    def __init__(self, tol: float = TOL, max_iter: int = MAX_ITER) -> None:
        """
        Initialize the MyPerceptron classifier.

        Args:
            tol (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.
        """
        self.tol: float = tol
        self.max_iter: int = max_iter
        self.sep_hp: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Train the perceptron model on the given dataset.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,), expected to be 1 or -1.

        Returns:
            np.ndarray: Learned weight vector representing the separating hyperplane.
        """
        logging.info("Starting training using fit method.")

        try:
            X = X - np.mean(X, axis=0)
            n, m = X.shape
            w_prime = np.mean(X, axis=0)
            w = np.zeros_like(w_prime)
            current_iter = 0

            while (np.linalg.norm(w_prime - w) > self.tol) and (current_iter < self.max_iter):
                w = w_prime.copy()
                ix = np.random.randint(n)
                v = X[ix]
                condition = np.dot(w, v.T)

                if y[ix] == 1:
                    if condition > 0:
                        current_iter += 1
                        continue
                    else:
                        w_prime = w + v
                elif y[ix] == -1:
                    if condition < 0:
                        current_iter += 1
                        continue
                    else:
                        w_prime = w - v

                current_iter += 1

            self.sep_hp = w_prime
            logging.info("Training completed successfully.")
            return w_prime

        except Exception as e:
            logging.exception("Exception occurred in fit()")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for given samples.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels (-1 or 1).
        """
        logging.info("Starting prediction using predict method.")

        try:
            if self.sep_hp is None:
                raise ValueError("Model is not trained yet. Call `fit()` first.")

            X = X - np.mean(X, axis=0)
            predictions = np.sign(np.dot(X, self.sep_hp))
            logging.info("Prediction completed successfully.")
            return predictions

        except Exception as e:
            logging.exception("Exception occurred in predict()")
            raise

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy of the model on given test data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).

        Returns:
            float: Classification accuracy as a float between 0 and 1.
        """
        logging.info("Starting accuracy calculation using accuracy method.")

        try:
            y_pred = self.predict(X)
            acc = np.mean(y == y_pred)
            logging.info("Accuracy calculation completed: %.4f", acc)
            return acc

        except Exception as e:
            logging.exception("Exception occurred in accuracy()")
            raise

