"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functionality to plot a confusion matrix using matplotlib,
    with support for normalization, error handling, and logging.
Version: 1.0
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import itertools
from typing import List, Union


# Ensure 'outputs' directory exists for logging
os.makedirs("outputs", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename=os.path.join("outputs", "plot.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ConfusionMatrixPlotter:
    """Class for plotting a confusion matrix with optional normalization and logging."""

    @staticmethod
    def plot(
        cm: np.ndarray,
        classes: List[str],
        normalize: bool = False,
        title: str = 'Confusion matrix',
        cmap: Union[str, None] = plt.cm.Blues
    ) -> None:
        """
        Plot a confusion matrix using matplotlib.

        Args:
            cm (np.ndarray): Confusion matrix (2D array).
            classes (List[str]): List of class names for labeling the axes.
            normalize (bool, optional): Whether to normalize the values. Defaults to False.
            title (str, optional): Title for the plot. Defaults to 'Confusion matrix'.
            cmap (Union[str, None], optional): Color map to use for the plot. Defaults to plt.cm.Blues.

        Raises:
            ValueError: If the confusion matrix is not a 2D square numpy array.
            Exception: For any unexpected errors during plotting.
        """
        try:
            if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
                logging.error("Confusion matrix must be a square 2D numpy array.")
                raise ValueError("Confusion matrix must be a square 2D numpy array.")

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                logging.info("Normalized confusion matrix")

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.0

            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.show()
            logging.info("Confusion matrix plotted successfully")

        except ValueError as ve:
            logging.exception("ValueError occurred while plotting confusion matrix.")
            raise ve
        except Exception as e:
            logging.exception("An unexpected error occurred while plotting confusion matrix.")
            raise e

