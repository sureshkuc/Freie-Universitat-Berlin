"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a LinearSVM class that implements hard-margin and soft-margin
    support vector machine training, along with classification methods.
    The module uses quadratic programming solvers from qpsolvers and cvxopt.
Version: 1.0
"""

import os
import logging
import numpy as np
from cvxopt import matrix, solvers as cvx_solver
import qpsolvers
import numpy.matlib
from typing import Tuple

# Create 'outputs' directory for logs if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Setup logging
log_file = os.path.join("outputs", "model.log")
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


class LinearSVM:
    """
    A class that implements linear Support Vector Machines with hard-margin
    and soft-margin training using quadratic programming.
    """

    @staticmethod
    def train_hard(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Train a hard-margin SVM classifier.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Label vector of shape (n_samples,) with values in {-1, 1}.

        Returns:
            Tuple[np.ndarray, float]: Weight vector w and bias term b.

        Raises:
            Exception: If training fails.
        """
        try:
            nsamples, ndims = X.shape
            P = np.eye(ndims + 1)
            P[ndims, ndims] = 1e-3  # small regularization for bias term
            q = np.zeros(ndims + 1)

            G = -np.hstack([
                np.multiply(np.matlib.repmat(y, ndims, 1).T, X),
                np.array(y).reshape(-1, 1)
            ])
            h = -np.ones(nsamples)

            z = qpsolvers.solve_qp(P, q, G, h)
            if z is None:
                logger.error("QP solver failed to find a solution in train_hard.")
                raise ValueError("QP solver failed to find a solution.")

            w, b = z[:ndims], z[ndims]
            logger.info("Hard-margin SVM trained successfully.")
            return w, b
        except Exception as e:
            logger.exception("Exception occurred in train_hard.")
            raise

    @staticmethod
    def train_soft(X: np.ndarray, y: np.ndarray, C: float) -> Tuple[np.ndarray, float]:
        """
        Train a soft-margin SVM classifier.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Label vector of shape (n_samples,) with values in {-1, 1}.
            C (float): Regularization parameter.

        Returns:
            Tuple[np.ndarray, float]: Weight vector w and average bias term b.

        Raises:
            Exception: If training fails.
        """
        try:
            m, n = X.shape
            y = y.reshape(-1, 1).astype(float)
            X_dash = y * X
            H = np.dot(X_dash, X_dash.T)

            P = matrix(H)
            q = matrix(-np.ones((m, 1)))
            G = matrix(np.vstack((-np.eye(m), np.eye(m))))
            h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
            A = matrix(y.reshape(1, -1))
            b = matrix(np.zeros(1))

            sol = cvx_solver.qp(P, q, G, h, A, b)
            a = np.array(sol['x'])

            w = ((y * a).T @ X).reshape(-1, 1)
            S = (a > 1e-6).flatten()
            b_values = y[S] - np.dot(X[S], w)
            b_mean = b_values.mean()

            logger.info("Soft-margin SVM trained successfully.")
            return w, b_mean
        except Exception as e:
            logger.exception("Exception occurred in train_soft.")
            raise

    @staticmethod
    def classify(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
        """
        Classify input samples using the trained SVM model.

        Args:
            w (np.ndarray): Weight vector of shape (n_features, 1) or (n_features,).
            b (float): Bias term.
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels of shape (n_samples,) with values in {-1, 1}.

        Raises:
            Exception: If classification fails.
        """
        try:
            r = np.dot(w.flatten(), X.T) + b
            predictions = np.sign(r)
            logger.info("Classification completed.")
            return predictions
        except Exception as e:
            logger.exception("Exception occurred in classify.")
            raise

