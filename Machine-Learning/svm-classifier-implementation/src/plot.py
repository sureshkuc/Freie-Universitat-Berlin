"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to visualize results from classification models,
    including decision boundaries and confusion matrices.
Version: 1.0
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import List, Union
from model import LinearSVM  # Assuming classify is a @staticmethod in LinearSVM

# Ensure outputs directory exists for logs
os.makedirs("outputs", exist_ok=True)

# Configure logging
log_file = os.path.join("outputs", "plot.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def plot_decision_boundary(
    X: np.ndarray,
    y: Union[np.ndarray, List[int]],
    w: np.ndarray,
    b: float,
    title: str = "Decision Boundary"
) -> None:
    """
    Plot the decision boundary for a 2D dataset using a linear classifier.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, 2).
        y (Union[np.ndarray, List[int]]): Class labels for each sample.
        w (np.ndarray): Weight vector for the decision boundary.
        b (float): Bias term.
        title (str): Title of the plot.

    Returns:
        None
    """
    try:
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        logger.debug("Meshgrid created for plotting decision boundary.")

        Z = LinearSVM.classify(w, b, np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title(title)
        plt.xlabel("Lightness")
        plt.ylabel("Size")
        plt.tight_layout()
        plt.show()

        logger.info("Decision boundary plot displayed successfully.")

    except Exception as e:
        logger.exception("Error occurred while plotting decision boundary.")


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot a confusion matrix using matplotlib.

    Args:
        cm (np.ndarray): Confusion matrix.
        classes (List[str]): List of class names.
        normalize (bool): Whether to normalize the values.
        title (str): Title of the plot.

    Returns:
        None
    """
    try:
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            logger.debug("Confusion matrix normalized.")

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()

        logger.info("Confusion matrix plot displayed successfully.")

    except Exception as e:
        logger.exception("Error occurred while plotting confusion matrix.")

