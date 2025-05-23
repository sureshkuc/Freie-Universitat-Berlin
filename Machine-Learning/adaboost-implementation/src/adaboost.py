"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module implements the AdaBoost ensemble classifier using
    decision stumps (weak learners) as base classifiers. It includes
    methods to fit the ensemble model and predict class labels.

Version: 1.0
"""

import os
import numpy as np
import logging
from typing import List, Tuple
from .decision_stump import DecisionTreeClassifier

# Setup logging
LOG_DIR = "outputs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'adaboost.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

class Adaboost:
    """
    AdaBoost classifier using decision stumps as weak learners.
    """

    def __init__(self, n_clf: int = 5, max_depth: int = 1) -> None:
        """
        Initializes the AdaBoost classifier.

        Args:
            n_clf (int): Number of weak classifiers to use.
            max_depth (int): Maximum depth of each weak classifier.
        """
        self.n_clf: int = n_clf
        self.max_depth: int = max_depth
        self.clfs: List[Tuple[DecisionTreeClassifier, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the AdaBoost model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Labels array of shape (n_samples,). Labels must be {-1, 1}.
        """
        try:
            n_samples = X.shape[0]
            w = np.full(n_samples, (1 / n_samples))  # Initialize weights

            for i in range(self.n_clf):
                clf = DecisionTreeClassifier(max_depth=self.max_depth).fit(X, y, w)
                preds = DecisionTreeClassifier()._predict_one(clf, X)
                error = np.sum(w[y != preds])
                
                # Avoid divide-by-zero
                error = np.clip(error, 1e-10, 1 - 1e-10)
                alpha = 0.5 * np.log((1.0 - error) / error)

                # Update weights
                w *= np.exp(-alpha * y * preds)
                w /= np.sum(w)

                self.clfs.append((clf, alpha))
                logging.info(f"Weak classifier {i+1}: error={error:.4f}, alpha={alpha:.4f}")

        except Exception as e:
            logging.error("Error in Adaboost.fit", exc_info=True)
            raise RuntimeError("Failed to fit Adaboost model.") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels of shape (n_samples,).
        """
        try:
            clf_preds = [alpha * DecisionTreeClassifier()._predict_one(clf, X)
                         for clf, alpha in self.clfs]
            final_pred = np.sign(np.sum(clf_preds, axis=0))
            return final_pred
        except Exception as e:
            logging.error("Error in Adaboost.predict", exc_info=True)
            raise RuntimeError("Failed to predict with Adaboost model.") from e

