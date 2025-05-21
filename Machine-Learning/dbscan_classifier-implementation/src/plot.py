"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to visualize clustering results,
    accuracy trends, and silhouette scores using matplotlib. It includes
    error handling and logs all events to the 'outputs' directory.
Version: 1.0
"""

import os
import logging
from typing import List
import matplotlib.pyplot as plt
import numpy as np

# Ensure 'outputs' directory exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='outputs/plot.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Logs all levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
)


def cluster_plot(
    dataset: np.ndarray,
    labels: np.ndarray,
    eps: float,
    minPts: int,
    data_size: int,
    acc: float,
    sil_coef: float
) -> None:
    """
    Plot the clustered dataset and display clustering metrics.

    Args:
        dataset (np.ndarray): The dataset containing 2D points.
        labels (np.ndarray): Cluster labels for each point.
        eps (float): Epsilon value used in clustering (e.g., DBSCAN).
        minPts (int): Minimum number of points for forming a cluster.
        data_size (int): Number of samples in the dataset.
        acc (float): Clustering accuracy in percentage.
        sil_coef (float): Silhouette coefficient score.

    Returns:
        None
    """
    try:
        plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='viridis')
        plt.axis('equal')
        plt.title(
            f'Eps={eps}, minPts={minPts}, Data Size={data_size}, '
            f'Accuracy={acc:.2f}%, Silhouette={sil_coef:.2f}'
        )
        plt.show()
        logging.info("Cluster plot generated successfully.")
    except Exception as e:
        logging.error("Error in plotting cluster: %s", str(e), exc_info=True)


def plot_accuracy(data_size_list: List[int], acc_list: List[float]) -> None:
    """
    Plot accuracy as a function of dataset size.

    Args:
        data_size_list (List[int]): List of dataset sizes.
        acc_list (List[float]): Corresponding accuracy values.

    Returns:
        None
    """
    try:
        plt.plot(data_size_list, acc_list, marker='o')
        plt.title('Accuracy vs Data Size')
        plt.xlabel('Data Size')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logging.info("Accuracy plot generated successfully.")
    except Exception as e:
        logging.error("Error in accuracy plot: %s", str(e), exc_info=True)


def plot_silhouette(data_size_list: List[int], sil_scores: List[float]) -> None:
    """
    Plot silhouette scores as a function of dataset size.

    Args:
        data_size_list (List[int]): List of dataset sizes.
        sil_scores (List[float]): Corresponding silhouette scores.

    Returns:
        None
    """
    try:
        plt.plot(data_size_list, sil_scores, marker='s', color='green')
        plt.title('Silhouette Score vs Data Size')
        plt.xlabel('Data Size')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logging.info("Silhouette plot generated successfully.")
    except Exception as e:
        logging.error("Error in silhouette plot: %s", str(e), exc_info=True)

