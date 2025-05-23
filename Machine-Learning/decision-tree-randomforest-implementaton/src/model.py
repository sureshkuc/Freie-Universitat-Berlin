"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides classes for Decision Tree and Random Forest classifiers,
    including training and prediction logic with support for Gini-based splitting.

Version: 1.0
"""

import os
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union

from .utils import to_terminal, gini_index
from .train import find_split
from .utils import bootstrap_sample

# Setup logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/model.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DecisionTreeClassifier:
    """
    A basic Decision Tree classifier using Gini index for splitting.
    """

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 1) -> None:
        """
        Initialize the DecisionTreeClassifier.

        Args:
            max_depth (int): Maximum depth of the tree.
            min_samples_leaf (int): Minimum samples required to be at a leaf node.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree: Optional[Dict[str, Any]] = None

    def fit(self, x: np.ndarray, y: np.ndarray, par_node: Optional[Dict[str, Any]] = None, depth: int = 0) -> Dict[str, Any]:
        """
        Train the decision tree on the given dataset.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            par_node (dict, optional): Partial tree node for recursion.
            depth (int): Current depth of the tree.

        Returns:
            dict: Tree structure as nested dictionary.
        """
        if par_node is None:
            par_node = {}

        try:
            cutoff, col, gini_gain, best_tree_childs = find_split(x, y)
            par_node = {
                'col': f'X{col}',
                'index_col': col,
                'cutoff': cutoff,
                'gini_gain': gini_gain
            }
            y_left, y_right = best_tree_childs

            if y_left.size == 0 or y_right.size == 0:
                terminal_value = to_terminal(np.concatenate((y_left, y_right)))
                par_node['left'] = terminal_value
                par_node['right'] = terminal_value
                return par_node

            if depth >= self.max_depth:
                par_node['left'] = to_terminal(y_left)
                par_node['right'] = to_terminal(y_right)
                return par_node

            if y_left.size <= self.min_samples_leaf:
                par_node['left'] = to_terminal(y_left)
            else:
                par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth + 1)

            if y_right.size <= self.min_samples_leaf:
                par_node['right'] = to_terminal(y_right)
            else:
                par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth + 1)

            self.tree = par_node
            return par_node

        except Exception as e:
            logger.exception("Error fitting decision tree")
            raise

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input samples.

        Args:
            x (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        try:
            return np.array([self._predict_single(self.tree, row) for row in x])
        except Exception as e:
            logger.exception("Prediction failed")
            raise

    def _predict_single(self, node: Union[Dict[str, Any], int], row: np.ndarray) -> int:
        """
        Predict the label for a single sample.

        Args:
            node (dict or int): Current tree node or terminal value.
            row (np.ndarray): Feature row.

        Returns:
            int: Predicted class label.
        """
        while isinstance(node, dict):
            if row[node['index_col']] <= node['cutoff']:
                node = node['left']
            else:
                node = node['right']
        return node


class RandomForestClassifier:
    """
    A basic implementation of Random Forest classifier using bagging and decision trees.
    """

    def __init__(self, n_estimators: int = 10, max_depth: int = 5, min_size: int = 10) -> None:
        """
        Initialize the RandomForestClassifier.

        Args:
            n_estimators (int): Number of trees in the forest.
            max_depth (int): Maximum depth of each tree.
            min_size (int): Minimum samples per leaf node.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_size = min_size
        self.trees: List[DecisionTreeClassifier] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Random Forest on training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        try:
            self.trees = []
            for _ in range(self.n_estimators):
                sample_X, sample_y = bootstrap_sample(X, y)
                tree = DecisionTreeClassifier(self.max_depth, self.min_size)
                tree.fit(sample_X, sample_y)
                self.trees.append(tree)
            logger.info("Random forest training completed.")
        except Exception as e:
            logger.exception("Error fitting random forest")
            raise

    def predict(self, row: np.ndarray) -> int:
        """
        Predict a label for a single sample using majority voting.

        Args:
            row (np.ndarray): Feature row.

        Returns:
            int: Predicted label.
        """
        try:
            predictions = [tree._predict_single(tree.tree, row) for tree in self.trees]
            return max(set(predictions), key=predictions.count)
        except Exception as e:
            logger.exception("Error during prediction")
            raise

    def predict_batch(self, X: np.ndarray) -> List[int]:
        """
        Predict labels for a batch of samples.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            List[int]: List of predicted labels.
        """
        try:
            return [self.predict(row) for row in X]
        except Exception as e:
            logger.exception("Batch prediction failed")
            raise

