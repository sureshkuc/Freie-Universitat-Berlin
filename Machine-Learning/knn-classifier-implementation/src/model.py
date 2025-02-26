"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a base classifier and a K-Nearest Neighbors (KNN) implementation.
    It includes methods for accuracy calculation, confusion matrix computation, and KNN prediction.
Version: 1.0
"""

import os
import logging
import numpy as np
from typing import Tuple, List

# Ensure the 'outputs' folder exists for logging
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

class Classifier:
    """Base classifier with utility functions for evaluation."""

    def accuracy(self, labels: np.ndarray, predictions: np.ndarray) -> float:
        """Computes the accuracy of predictions.

        Args:
            labels (np.ndarray): True labels.
            predictions (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        try:
            return np.mean(labels == predictions)
        except Exception as e:
            logging.error("Error calculating accuracy: %s", str(e))
            raise

    def confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Computes the confusion matrix.

        Args:
            labels (np.ndarray): True labels.
            predictions (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Confusion matrix.
        """
        try:
            size = len(np.unique(labels))
            matrix = np.zeros((size, size), dtype=int)
            for correct, predicted in zip(labels.astype(int), predictions.astype(int)):
                matrix[correct, predicted] += 1
            return matrix
        except Exception as e:
            logging.error("Error computing confusion matrix: %s", str(e))
            raise

class KNearestNeighbors(Classifier):
    """K-Nearest Neighbors classifier implementation."""

    def __init__(self) -> None:
        """Initializes KNearestNeighbors class."""
        self.X = None
        self.y = None

    def euclidean_distance(self, x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        """Calculates Euclidean distance between two points.

        Args:
            x_1 (np.ndarray): First point or array of points.
            x_2 (np.ndarray): Second point.

        Returns:
            np.ndarray: Computed distances.
        """
        try:
            return np.sqrt(np.sum((x_1 - x_2) ** 2, axis=1))
        except Exception as e:
            logging.error("Error calculating Euclidean distance: %s", str(e))
            raise

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stores the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        try:
            self.X = X
            self.y = y
        except Exception as e:
            logging.error("Error during model fitting: %s", str(e))
            raise

    def predict(self, X_test: np.ndarray, k: int) -> np.ndarray:
        """Predicts labels using the k-nearest neighbors algorithm.

        Args:
            X_test (np.ndarray): Test data.
            k (int): Number of neighbors to consider.

        Returns:
            np.ndarray: Predicted labels.
        """
        try:
            predictions = []
            for sample in X_test:
                distances = self.euclidean_distance(self.X, sample)
                indices = np.argpartition(distances, k)[:k]
                votes = self.y[indices].astype(int)
                winner = np.argmax(np.bincount(votes))
                predictions.append(winner)
            logging.info("Predictions completed for k=%d", k)
            return np.array(predictions)
        except Exception as e:
            logging.error("Error during prediction: %s", str(e))
            raise

if __name__ == "__main__":
    """Main execution block for testing the KNN classifier."""
    try:
        # Example dataset
        X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 1, 0])
        X_test = np.array([[2, 2], [3, 5]])
        
        knn = KNearestNeighbors()
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test, k=2)
        
        print("Predictions:", predictions)
    except Exception as e:
        logging.critical("Critical error in main execution: %s", str(e))

