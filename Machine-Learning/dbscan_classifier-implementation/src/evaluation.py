"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a utility function to evaluate clustering performance
    using accuracy and silhouette score. It handles missing predictions and
    provides basic logging of errors during evaluation.

Version: 1.0
"""

import os
import logging
from typing import Tuple, List, Union

import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join("outputs", "clustering_evaluation.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_clustering(
    true_labels: Union[List[int], np.ndarray],
    pred_labels: Union[List[Union[int, None]], np.ndarray],
    dataset: Union[List[List[float]], np.ndarray]
) -> Tuple[float, float]:
    """
    Evaluate the clustering performance using accuracy and silhouette score.

    Args:
        true_labels (List[int] or np.ndarray): Ground truth cluster labels.
        pred_labels (List[int or None] or np.ndarray): Predicted cluster labels. `None` will be treated as -1.
        dataset (List[List[float]] or np.ndarray): The input dataset used for clustering.

    Returns:
        Tuple[float, float]: A tuple containing:
            - Accuracy as a percentage (0–100).
            - Silhouette score (-1 if not computable due to cluster count).
    """
    try:
        # Replace None with -1 in predicted labels
        pred_labels = [x if x is not None else -1 for x in pred_labels]

        # Ensure arrays are NumPy arrays for consistency
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        dataset = np.array(dataset)

        correct = np.sum(true_labels == pred_labels)
        acc = (correct * 100) / len(dataset)

        n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)

        if n_clusters > 1:
            sil_score = metrics.silhouette_score(dataset, pred_labels)
        else:
            sil_score = -1.0

        logging.info("Clustering evaluated successfully. Accuracy: %.2f%%, Silhouette Score: %.4f", acc, sil_score)
        return acc, sil_score

    except Exception as e:
        logging.error("Error in evaluating clustering: %s", str(e), exc_info=True)
        return 0.0, -1.0

