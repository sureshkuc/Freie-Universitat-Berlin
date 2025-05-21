"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module generates synthetic two-spiral data, applies DBSCAN clustering,
    evaluates performance using accuracy and silhouette score metrics, and visualizes results.
    It includes hyperparameter tuning and plots performance across different dataset sizes.
Version: 1.0
"""

import os
import numpy as np
from typing import Tuple, List

from model import DBSCANClustering
from evaluation import evaluate_clustering
from plot import cluster_plot, plot_accuracy, plot_silhouette
from config import logging


# Ensure logging writes all levels to outputs folder
log_dir = "outputs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "experiment.log")

# Update logging configuration
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)


def twospirals(n_points: int, noise: float = 0.5) -> np.ndarray:
    """
    Generate a two-spiral dataset.

    Args:
        n_points (int): Number of points per spiral.
        noise (float): Noise level to be added to the spirals.

    Returns:
        np.ndarray: A (2 * n_points, 2) array containing the dataset.
    """
    epsilon = 0.1
    n = (np.random.rand(n_points, 1) + epsilon) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    c_1 = np.hstack((d1x, d1y))
    c_2 = np.hstack((-d1x, -d1y))
    return np.vstack((c_1, c_2))


def create_dataset(data_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic two-spiral dataset with labels.

    Args:
        data_size (int): Number of samples per spiral.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Dataset and corresponding labels.
    """
    data = twospirals(data_size)
    labels = np.hstack((np.zeros(data_size), np.ones(data_size)))
    return data, labels


def tune_hyperparameters() -> Tuple[float, int, float, float]:
    """
    Perform a grid search over epsilon and minPts to find the best DBSCAN parameters.

    Returns:
        Tuple[float, int, float, float]: Best epsilon, minPts, accuracy, and silhouette score.
    """
    try:
        eps_list = np.arange(0.1, 2.0, 0.05)
        min_pts_list = np.arange(2, 10, 1)
        results = []

        for min_pts in min_pts_list:
            for eps in eps_list:
                np.random.seed(10)
                data, true_labels = create_dataset(500)
                pred_labels = DBSCANClustering.dbscan(data, eps, min_pts)
                acc, sil = evaluate_clustering(true_labels, pred_labels, data)
                results.append((eps, min_pts, acc, sil))

        best_result = sorted(results, key=lambda x: (x[2], x[3]), reverse=True)[0]
        logging.info(f"Best hyperparameters found: eps={best_result[0]}, minPts={best_result[1]}, "
                     f"Accuracy={best_result[2]:.4f}, Silhouette={best_result[3]:.4f}")
        return best_result
    except Exception as error:
        logging.error("Error in tuning hyperparameters: %s", str(error), exc_info=True)
        raise


def run_experiment() -> None:
    """
    Run the DBSCAN clustering experiment with hyperparameter tuning and evaluation across data sizes.
    """
    try:
        best_eps, best_min_pts, best_acc, best_sil = tune_hyperparameters()
        np.random.seed(10)
        dataset, labels_true = create_dataset(500)
        labels = DBSCANClustering.dbscan(dataset, best_eps, best_min_pts)
        cluster_plot(dataset, labels, best_eps, best_min_pts, 500, best_acc, best_sil)

        data_sizes = range(50, 2000, 25)
        acc_list: List[float] = []
        sil_list: List[float] = []

        for size in data_sizes:
            np.random.seed(10)
            dataset, labels_true = create_dataset(size)
            labels = DBSCANClustering.dbscan(dataset, best_eps, best_min_pts)
            acc, sil = evaluate_clustering(labels_true, labels, dataset)
            acc_list.append(acc)
            sil_list.append(sil)
            cluster_plot(dataset, labels, best_eps, best_min_pts, size, acc, sil)

        plot_accuracy(data_sizes, acc_list)
        plot_silhouette(data_sizes, sil_list)

        for size, acc in zip(data_sizes, acc_list):
            print(f"Data Size: {size}, Accuracy: {acc:.2f}%")

        logging.info("Experiment completed successfully.")

    except Exception as error:
        logging.critical("Fatal error in main experiment: %s", str(error), exc_info=True)


if __name__ == "__main__":
    run_experiment()

