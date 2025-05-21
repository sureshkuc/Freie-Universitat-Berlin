"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to evaluate a binary classification model.
    It includes computation of F1-score, classification report, and ROC curve plotting.
Version: 1.0
"""

import os
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve


# Ensure the 'outputs' folder exists for logs
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join("outputs", "evaluation.log"),
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_model(y_test: np.ndarray, predictions: np.ndarray) -> Tuple[float, str]:
    """
    Evaluate the classification performance using F1-score and generate a classification report.

    Args:
        y_test (np.ndarray): Ground truth binary labels.
        predictions (np.ndarray): Predicted probabilities from the model.

    Returns:
        Tuple[float, str]: F1-score and classification report string.
    """
    try:
        logging.debug("Evaluating model...")
        y_pred_binary = np.where(predictions > 0.5, 1, 0)
        score = f1_score(y_test, y_pred_binary)
        report = classification_report(y_test, y_pred_binary, target_names=['digit 1', 'digit 9'])
        logging.info("Model evaluation successful.")
        return score, report
    except Exception as e:
        logging.exception("Error during model evaluation.")
        raise


def plot_roc_curve(y_test: np.ndarray, predictions: np.ndarray) -> None:
    """
    Plot the ROC curve for the given ground truth and predicted probabilities.

    Args:
        y_test (np.ndarray): Ground truth binary labels.
        predictions (np.ndarray): Predicted probabilities from the model.

    Returns:
        None
    """
    try:
        logging.debug("Plotting ROC curve...")
        fpr, tpr, _ = roc_curve(y_test, predictions)
        auc = roc_auc_score(y_test, predictions)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        logging.info("ROC curve plotted successfully.")
    except Exception as e:
        logging.exception("Error while plotting ROC curve.")
        raise

