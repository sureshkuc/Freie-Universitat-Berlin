"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a class to evaluate classification models using sklearn's 
    classification report. It includes robust logging and error handling.
Version: 1.0
"""

import os
import logging
from typing import Any
from sklearn.metrics import classification_report


class ModelEvaluator:
    """
    A class for evaluating classification models.

    This class provides a method to generate a classification report
    and logs errors appropriately.
    """

    def __init__(self) -> None:
        """
        Initializes the ModelEvaluator class and configures logging.
        """
        self._setup_logging()

    def _setup_logging(self) -> None:
        """
        Configures logging to store logs in the 'outputs' folder.
        """
        os.makedirs("outputs", exist_ok=True)
        logging.basicConfig(
            filename="outputs/evaluation.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.debug("Logging initialized for ModelEvaluator.")

    def evaluate(self, y_true: Any, y_pred: Any) -> str:
        """
        Evaluates the model and returns the classification report.

        Args:
            y_true (Any): Ground truth (correct) target values.
            y_pred (Any): Estimated target values as predicted by the model.

        Returns:
            str: The classification report.

        Raises:
            Exception: Any exception raised during the evaluation process is logged and re-raised.
        """
        try:
            report = classification_report(y_true, y_pred)
            logging.info("Model evaluation successful.")
            return report
        except Exception as e:
            logging.exception("Error occurred in evaluate method.")
            raise


# Example usage (only run this block if the script is executed directly)
evaluator = ModelEvaluator()
y_true_example = [1, 0, 1, 1, 0]
y_pred_example = [1, 0, 0, 1, 0]

try:
	result = evaluator.evaluate(y_true_example, y_pred_example)
	print(result)
except Exception as err:
	print(f"An error occurred during evaluation: {err}")

