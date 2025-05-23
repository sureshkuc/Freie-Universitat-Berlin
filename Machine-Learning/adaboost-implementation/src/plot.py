"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to visualize model evaluation results,
    including plotting confusion matrices and generating classification reports.
Version: 1.0
"""

import os
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Union

# Ensure logging directory exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/plot.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

class ModelVisualizer:
    """
    A class for visualizing the results of a classification model.
    Includes confusion matrix plotting and classification report generation.
    """

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        classes: List[str],
        normalize: bool = False,
        title: str = "Confusion Matrix",
        cmap: Union[str, plt.Colormap] = plt.cm.Blues
    ) -> None:
        """
        Plots the confusion matrix.

        Args:
            cm (np.ndarray): Confusion matrix to be visualized.
            classes (List[str]): List of class names.
            normalize (bool): Whether to normalize the matrix.
            title (str): Title of the plot.
            cmap (Union[str, plt.Colormap]): Color map for the matrix.

        Raises:
            Exception: If plotting fails.
        """
        try:
            logging.debug("Starting plot_confusion_matrix")

            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                logging.info("Normalized confusion matrix.")

            plt.imshow(cm, interpolation="nearest", cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = ".2f" if normalize else "d"
            thresh = cm.max() / 2.0
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.show()
            logging.info("Confusion matrix plotted successfully.")
        except Exception as e:
            logging.exception("Error in plot_confusion_matrix")
            raise

    @staticmethod
    def plot_model_report(y_test: List[int], y_pred: List[int], labels: List[str] = None) -> None:
        """
        Prints classification report and plots confusion matrix.

        Args:
            y_test (List[int]): True labels.
            y_pred (List[int]): Predicted labels.
            labels (List[str], optional): Label names. Defaults to ['Not Spam', 'Spam'].

        Raises:
            Exception: If report generation or plotting fails.
        """
        try:
            logging.debug("Starting plot_model_report")

            if labels is None:
                labels = ["Not Spam", "Spam"]

            report = classification_report(y_test, y_pred, target_names=labels)
            print(report)
            logging.info("Classification report:\n%s", report)

            cm = confusion_matrix(y_test, y_pred)
            logging.debug(f"Confusion Matrix:\n{cm}")
            ModelVisualizer.plot_confusion_matrix(cm, classes=labels)
        except Exception as e:
            logging.exception("Error in plot_model_report")
            raise

