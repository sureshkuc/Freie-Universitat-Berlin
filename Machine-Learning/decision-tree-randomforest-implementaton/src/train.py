"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a class to handle the logic for finding the best feature split
    in a dataset using Gini impurity for decision tree construction.
Version: 1.0
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional
from .utils import to_terminal, gini_index
from .config import RANDOM_STATE

# Setup logging
LOG_DIR = 'outputs'
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'train.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DecisionTreeSplitter:
    """
    A class to find the optimal feature split for a decision tree node using Gini impurity.
    """

    def __init__(self, random_state: Optional[int] = RANDOM_STATE) -> None:
        """
        Initialize the splitter.

        Args:
            random_state (Optional[int]): Seed for reproducibility.
        """
        self.random_state = random_state

    def find_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[float], Optional[int], float, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Find the best split for the dataset based on Gini impurity.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.

        Returns:
            Tuple containing:
                - best_cutoff (Optional[float]): The value to split the feature on.
                - best_col (Optional[int]): The index of the feature column.
                - gini_gain (float): The gain in Gini impurity after the split.
                - best_children (Optional[Tuple[np.ndarray, np.ndarray]]): Tuple of left and right child labels.
        """
        try:
            class_values = np.unique(y)
            min_gini = float('inf')
            best_col = None
            best_cutoff = None
            best_children = None
            gini_total = 0.0

            for col in range(X.shape[1]):
                unique_vals = np.unique(X[:, col])
                for cutoff in unique_vals:
                    y_left = y[X[:, col] < cutoff]
                    y_right = y[X[:, col] >= cutoff]
                    gini = gini_index((y_left, y_right), class_values)

                    if gini < min_gini:
                        min_gini = gini
                        best_col = col
                        best_cutoff = cutoff
                        best_children = (y_left, y_right)

            for val in class_values:
                p = np.mean(y == val)
                gini_total += p * p
            gini_parent = 1.0 - gini_total
            gini_gain = (gini_parent - min_gini) * y.size / y.size

            logging.info(
                f"Best split found at column {best_col} with cutoff {best_cutoff}, Gini gain: {gini_gain:.4f}"
            )
            return best_cutoff, best_col, gini_gain, best_children

        except Exception as e:
            logging.error("Error in find_split", exc_info=True)
            return None, None, 0.0, None

