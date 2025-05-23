"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides evaluation functionality for classification models,
    including performance metrics and confusion matrix plotting.
Version: 1.0
"""

import os
import logging
from typing import List, Union

from sklearn.metrics import classification_report, confusion_matrix
from .plot import plot_confusion_matrix


# Setup logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/evaluation.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


class ModelEvaluator:
    """
    A class to evaluate classification models using scikit-learn metrics and confusion matrix visualization.
    """

    def __init__(self) -> None:
        """
        Initializes the evaluator with default label names.
        """
        self.labels = ['Not Spam', 'Spam']
        logging.debug("ModelEvaluator initialized with labels: %s", self.labels)

    def evaluate(self, y_true: List[Union[int, bool]], y_pred: List[Union[int, bool]]) -> None:
        """
        Evaluate the model's predictions and print a classification report
        along with a confusion matrix.

        Args:
            y_true (List[Union[int, bool]]): The ground truth labels.
            y_pred (List[Union[int, bool]]): The predicted labels.
        """
        try:
            logging.info("Starting evaluation of the model.")
            report = classification_report(y_true, y_pred, target_names=self.labels)
            print(report)
            logging.info("Classification Report:\n%s", report)

            cm = confusion_matrix(y_true, y_pred)
            logging.debug("Confusion matrix calculated: %s", cm.tolist())

            plot_confusion_matrix(cm, self.labels, title='Confusion Matrix')
            logging.info("Confusion matrix plotted successfully.")

        except Exception as e:
            logging.error("Error during model evaluation: %s", str(e), exc_info=True)
            print("An error occurred during evaluation. Please check logs for details.")
evaluator = ModelEvaluator()
evaluator.evaluate([0, 1, 1, 0], [0, 0, 1, 1])  # Example usage

