"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to evaluate machine learning models using 
    common classification metrics such as accuracy, F1-score, classification 
    report, and confusion matrix. All errors and logs are written to the 'outputs' 
    folder for diagnostics and debugging.

Version: 1.0
"""

import os
import logging
from typing import Tuple, List, Any

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Ensure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Set up logging
logging.basicConfig(
    filename="outputs/evaluation.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true: List[Any], y_pred: List[Any]) -> Tuple[float, float]:
    """
    Evaluates the model using accuracy and F1 score.

    Args:
        y_true (List[Any]): The ground truth target values.
        y_pred (List[Any]): The predicted target values by the model.

    Returns:
        Tuple[float, float]: A tuple containing accuracy and F1 score.

    Raises:
        Exception: If evaluation fails due to input issues or metric computation errors.
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        logger.info("Model evaluation successful. Accuracy: %.4f, F1 Score: %.4f", acc, f1)
        return acc, f1
    except Exception as e:
        logger.exception("Evaluation failed due to an exception.")
        raise


def get_classification_report(
    y_true: List[Any], y_pred: List[Any], target_names: List[str]
) -> str:
    """
    Generates a detailed classification report.

    Args:
        y_true (List[Any]): The ground truth target values.
        y_pred (List[Any]): The predicted target values by the model.
        target_names (List[str]): Names of the target classes.

    Returns:
        str: Text summary of the precision, recall, F1 score, and support.
    """
    try:
        report = classification_report(y_true, y_pred, target_names=target_names)
        logger.info("Classification report generated successfully.")
        return report
    except Exception as e:
        logger.exception("Failed to generate classification report.")
        raise


def get_confusion_matrix(y_true: List[Any], y_pred: List[Any]) -> Any:
    """
    Computes the confusion matrix.

    Args:
        y_true (List[Any]): The ground truth target values.
        y_pred (List[Any]): The predicted target values by the model.

    Returns:
        Any: Confusion matrix as a 2D array.

    Raises:
        Exception: If confusion matrix computation fails.
    """
    try:
        matrix = confusion_matrix(y_true, y_pred)
        logger.info("Confusion matrix computed successfully.")
        return matrix
    except Exception as e:
        logger.exception("Failed to compute confusion matrix.")
        raise

