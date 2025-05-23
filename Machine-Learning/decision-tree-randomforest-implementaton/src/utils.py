"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides utility functions for machine learning tasks,
    including Gini index calculation, terminal value estimation,
    and bootstrapped sampling.
Version: 1.0
"""

import os
import logging
import random
import numpy as np
from typing import List, Tuple, Any

# Create logging directory
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/utils.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class MLUtils:
    """
    A utility class for machine learning-related helper functions.
    """

    @staticmethod
    def to_terminal(y: np.ndarray) -> float:
        """
        Calculate the terminal node value as the rounded mean.

        Args:
            y (np.ndarray): Target values.

        Returns:
            float: Rounded mean value.
        """
        try:
            result = float(np.round(np.mean(y)))
            logging.debug(f"Terminal value calculated: {result}")
            return result
        except Exception as e:
            logging.error("Error calculating terminal value", exc_info=True)
            raise

    @staticmethod
    def gini_index(groups: List[np.ndarray], classes: List[Any]) -> float:
        """
        Compute the Gini index for a split dataset.

        Args:
            groups (List[np.ndarray]): List of arrays with group data.
            classes (List[Any]): List of unique class labels.

        Returns:
            float: Gini index.
        """
        try:
            n_instances = float(sum([group.size for group in groups]))
            if n_instances == 0:
                logging.warning("No instances found when calculating Gini index.")
                return 0.0

            gini = 0.0
            for group in groups:
                size = group.size
                if size == 0:
                    continue
                score = 0.0
                for class_val in classes:
                    p = np.sum(group == class_val) / size
                    score += p * p
                group_gini = (1.0 - score) * (size / n_instances)
                gini += group_gini
                logging.debug(f"Gini for group: {group_gini}")

            logging.info(f"Total Gini index: {gini}")
            return gini
        except Exception as e:
            logging.error("Error calculating Gini index", exc_info=True)
            raise

    @staticmethod
    def bootstrap_sample(X: List[Any], y: List[Any]) -> Tuple[List[Any], List[Any]]:
        """
        Generate a bootstrap sample from the dataset.

        Args:
            X (List[Any]): Feature list.
            y (List[Any]): Label list.

        Returns:
            Tuple[List[Any], List[Any]]: Bootstrapped samples of features and labels.
        """
        try:
            n = len(X)
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]
            logging.debug(f"Bootstrap indices: {indices}")
            return X_sample, y_sample
        except Exception as e:
            logging.error("Error generating bootstrap sample", exc_info=True)
            raise

