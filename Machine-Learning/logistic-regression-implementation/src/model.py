"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a custom implementation of the Logistic Regression algorithm using
    cross-entropy and mean squared error loss functions. It supports training with gradient descent
    and includes logging and error handling for robustness.
Version: 1.0
"""

import numpy as np
import logging
from typing import Tuple

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler('outputs/model.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class LogisticRegression:
    """Custom implementation of logistic regression classifier."""

    def __init__(self, w: np.ndarray, b: float) -> None:
        """
        Initialize the logistic regression model.

        Args:
            w (np.ndarray): Weight vector.
            b (float): Bias term.
        """
        self.w = w
        self.b = b

    def sigmoid_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid activation function.

        Args:
            x (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Sigmoid probabilities.
        """
        try:
            return 1 / (1 + np.exp(-(np.dot(x, self.w) + self.b)))
        except Exception as e:
            logger.error("Error in sigmoid_function: %s", str(e))
            raise

    def cross_entropy_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cross-entropy loss.

        Args:
            x (np.ndarray): Input feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: Loss value.
        """
        try:
            sigmoid = self.sigmoid_function(x)
            return -1 / x.shape[0] * np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
        except Exception as e:
            logger.error("Error in cross_entropy_loss: %s", str(e))
            raise

    def mean_squared_error(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the mean squared error.

        Args:
            x (np.ndarray): Input feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: MSE value.
        """
        try:
            return np.mean((y - self.sigmoid_function(x)) ** 2)
        except Exception as e:
            logger.error("Error in mean_squared_error: %s", str(e))
            raise

    def gradient_descent(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients using cross-entropy loss.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): True labels.

        Returns:
            Tuple[np.ndarray, float]: Gradients for weights and bias.
        """
        try:
            predictions = self.sigmoid_function(x)
            dw = (1 / x.shape[0]) * np.dot(x.T, (predictions - y))
            db = (1 / x.shape[0]) * np.sum(predictions - y)
            return dw, db
        except Exception as e:
            logger.error("Error in gradient_descent: %s", str(e))
            raise

    def gradient_mse(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients using mean squared error.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): True labels.

        Returns:
            Tuple[np.ndarray, float]: Gradients for weights and bias.
        """
        try:
            pred = self.sigmoid_function(x)
            dw = (1 / x.shape[0]) * np.dot(x.T, pred * (1 - pred) * (pred - y))
            db = (1 / x.shape[0]) * np.sum(pred - y)
            return dw, db
        except Exception as e:
            logger.error("Error in gradient_mse: %s", str(e))
            raise

    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate: float, num_iter: int) -> "LogisticRegression":
        """
        Fit the model using cross-entropy loss.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            learning_rate (float): Learning rate.
            num_iter (int): Number of iterations.

        Returns:
            LogisticRegression: Trained model.
        """
        try:
            loss = [self.cross_entropy_loss(x, y)]
            for i in range(num_iter):
                dw, db = self.gradient_descent(x, y)
                self.w -= learning_rate * dw
                self.b -= learning_rate * db
                current_loss = self.cross_entropy_loss(x, y)
                loss.append(current_loss)
                if abs(loss[-2] - loss[-1]) < 0.001:
                    break
            return self
        except Exception as e:
            logger.error("Error in fit: %s", str(e))
            raise

    def fit_mse(self, x: np.ndarray, y: np.ndarray, learning_rate: float, num_iter: int) -> "LogisticRegression":
        """
        Fit the model using mean squared error.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            learning_rate (float): Learning rate.
            num_iter (int): Number of iterations.

        Returns:
            LogisticRegression: Trained model.
        """
        try:
            loss = [self.mean_squared_error(x, y)]
            for i in range(num_iter):
                dw, db = self.gradient_mse(x, y)
                self.w -= learning_rate * dw
                self.b -= learning_rate * db
                current_loss = self.mean_squared_error(x, y)
                loss.append(current_loss)
                if abs(loss[-2] - loss[-1]) < 0.001:
                    break
            return self
        except Exception as e:
            logger.error("Error in fit_mse: %s", str(e))
            raise

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the trained logistic regression model.

        Args:
            x (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        try:
            return self.sigmoid_function(x)
        except Exception as e:
            logger.error("Error in predict: %s", str(e))
            raise

