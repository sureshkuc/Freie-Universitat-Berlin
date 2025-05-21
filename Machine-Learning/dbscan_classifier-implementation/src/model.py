"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a class implementing the DBSCAN clustering algorithm.
    It includes methods for calculating Euclidean distance, generating a range
    query lookup table, and applying the DBSCAN logic to a dataset.
Version: 1.0
"""

import os
import logging
from typing import List, Dict, Set, Optional
import numpy as np

# Configure logging to file
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/dbscan_clustering.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class DBSCANClustering:
    """Class implementing DBSCAN clustering algorithm."""

    @staticmethod
    def euclidean_distance(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        """
        Calculate the Euclidean distance between a point and all points in the dataset.

        Args:
            x_1 (np.ndarray): A single data point.
            x_2 (np.ndarray): The dataset to compare against.

        Returns:
            np.ndarray: Array of distances.
        """
        return np.sqrt(np.sum((x_1 - x_2) ** 2, axis=1))

    @staticmethod
    def lookup_table(dataset: np.ndarray, eps: float) -> Dict[int, np.ndarray]:
        """
        Generate a lookup table mapping each point to its neighbors within eps distance.

        Args:
            dataset (np.ndarray): The dataset of points.
            eps (float): The epsilon threshold.

        Returns:
            Dict[int, np.ndarray]: Mapping of point index to indices of its neighbors.
        """
        lookup_2d_table = {}
        for pos, val in enumerate(dataset):
            dist = DBSCANClustering.euclidean_distance(val, dataset)
            neighbors = np.argwhere(dist <= eps).flatten()
            lookup_2d_table[pos] = neighbors
            logger.debug(f"Point {pos}: found {len(neighbors)} neighbors within eps={eps}")
        return lookup_2d_table

    @staticmethod
    def dbscan(data: np.ndarray, eps: float = 2.0, min_pts: int = 2) -> List[Optional[int]]:
        """
        Apply DBSCAN clustering algorithm to a dataset.

        Args:
            data (np.ndarray): Dataset of points.
            eps (float): Radius to consider for neighborhood.
            min_pts (int): Minimum number of points to form a dense region.

        Returns:
            List[Optional[int]]: List of cluster labels for each point (-1 for noise).
        """
        try:
            n_points = len(data)
            labels: List[Optional[int]] = [-1] * n_points
            cluster_id = -1
            visited: Set[int] = set()

            range_query = DBSCANClustering.lookup_table(data, eps)

            for point_idx in range(n_points):
                if labels[point_idx] != -1:
                    continue

                neighbors = range_query[point_idx]
                if len(neighbors) < min_pts:
                    labels[point_idx] = None  # Mark as noise
                    logger.info(f"Point {point_idx} marked as noise.")
                    continue

                cluster_id += 1
                labels[point_idx] = cluster_id
                seeds = set(neighbors)
                seeds.discard(point_idx)

                logger.info(f"Forming cluster {cluster_id} starting from point {point_idx}.")

                while seeds:
                    current_point = seeds.pop()

                    if labels[current_point] is None:
                        labels[current_point] = cluster_id

                    if labels[current_point] != -1:
                        continue

                    labels[current_point] = cluster_id
                    current_neighbors = range_query[current_point]

                    if len(current_neighbors) >= min_pts:
                        seeds.update(current_neighbors)

            logger.info("DBSCAN clustering completed successfully.")
            return labels

        except Exception as error:
            logger.error("Error in DBSCAN algorithm: %s", str(error))
            logger.debug("Detailed exception info", exc_info=True)
            raise

