"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a simple implementation of a Decision Tree Classifier
    using a recursive algorithm with a specified maximum depth and minimum
    samples for splitting. It is designed to handle weighted data and to split
    nodes using custom utility functions.

Version: 1.0
"""

import os
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Union

# Set up logging to "outputs" folder
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/decision_tree.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class DecisionTreeClassifier:
    """
    A decision tree classifier with maximum depth and minimum sample split constraints.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 1) -> None:
        """
        Initialize the DecisionTreeClassifier.

        Args:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum samples required to split a node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: Optional[Dict[str, Any]] = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        depth: int = 0
    ) -> Union[Dict[str, Any], Any]:
        """
        Fit the decision tree classifier to the data.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            w (np.ndarray): Sample weights.
            depth (int): Current depth of the tree.

        Returns:
            dict or any: The trained decision tree structure or terminal value.
        """
        try:
            from .tree_utils import find_split, to_terminal

            if depth >= self.max_depth or len(np.unique(y)) == 1:
                logging.info("Creating terminal node due to max depth or pure class.")
                return to_terminal(y, w)

            cutoff, col, _, best_tree_childs, best_weights = find_split(x, y, w)
            y_left, y_right = best_tree_childs
            w_left, w_right = best_weights

            node = {'index_col': col, 'cutoff': cutoff}
            logging.debug(f"Split at col={col}, cutoff={cutoff}")

            if len(y_left) <= self.min_samples_split:
                node['left'] = to_terminal(y_left, w_left)
            else:
                node['left'] = self.fit(x[x[:, col] < cutoff], y_left, w_left, depth + 1)

            if len(y_right) <= self.min_samples_split:
                node['right'] = to_terminal(y_right, w_right)
            else:
                node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, w_right, depth + 1)

            self.tree = node
            return node

        except Exception as e:
            logging.error(f"Error in fit method: {e}", exc_info=True)
            raise

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for given input features.

        Args:
            x (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        try:
            if self.tree is None:
                raise ValueError("Model has not been trained yet.")

            return np.array([self._predict_one(self.tree, row) for row in x])
        except Exception as e:
            logging.error(f"Error in predict method: {e}", exc_info=True)
            raise

    def _predict_one(self, node: Union[Dict[str, Any], Any], row: np.ndarray) -> Any:
        """
        Predict the label for a single input sample.

        Args:
            node (dict or any): Tree node or terminal value.
            row (np.ndarray): Input feature row.

        Returns:
            any: Predicted label.
        """
        try:
            while isinstance(node, dict):
                if row[node['index_col']] < node['cutoff']:
                    node = node['left']
                else:
                    node = node['right']
            return node
        except Exception as e:
            logging.error(f"Error in _predict_one method: {e}", exc_info=True)
            raise

