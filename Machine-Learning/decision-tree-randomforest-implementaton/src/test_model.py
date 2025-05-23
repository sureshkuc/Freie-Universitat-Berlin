"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module contains test cases for evaluating the functionality of 
    custom machine learning models including DecisionTreeClassifier, 
    RandomForestClassifier, utility functions, data splitting, and evaluation metrics.
Version: 1.0
"""

import logging
import os
from typing import List, Tuple

from model import DecisionTreeClassifier, RandomForestClassifier
from utils import gini_index, bootstrap_sample
from train import get_split
from evaluation import evaluate_model

# Set up logging
LOG_FILE = os.path.join("outputs", "test_log.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

def test_decision_tree() -> None:
    """
    Test the DecisionTreeClassifier's basic fit and predict functionality.
    """
    try:
        X: List[List[int]] = [[1], [2], [3], [4]]
        y: List[int] = [0, 0, 1, 1]
        model = DecisionTreeClassifier()
        model.fit(X, y)
        prediction = model.predict([1])
        assert prediction in [0, 1]
        logging.info("test_decision_tree passed.")
    except Exception as e:
        logging.error("test_decision_tree failed.", exc_info=True)
        raise e

def test_random_forest() -> None:
    """
    Test the RandomForestClassifier's fit and predict functionality.
    """
    try:
        X: List[List[int]] = [[1], [2], [3], [4]]
        y: List[int] = [0, 0, 1, 1]
        model = RandomForestClassifier(n_estimators=5)
        model.fit(X, y)
        prediction = model.predict([1])
        assert prediction in [0, 1]
        logging.info("test_random_forest passed.")
    except Exception as e:
        logging.error("test_random_forest failed.", exc_info=True)
        raise e

def test_gini() -> None:
    """
    Test the gini_index calculation.
    """
    try:
        groups = [[[1, 0]], [[2, 1], [3, 1]]]
        classes = [0, 1]
        gini = gini_index(groups, classes)
        assert 0 <= gini <= 1
        logging.info("test_gini passed.")
    except Exception as e:
        logging.error("test_gini failed.", exc_info=True)
        raise e

def test_bootstrap() -> None:
    """
    Test the bootstrap_sample function.
    """
    try:
        X = [[i] for i in range(10)]
        y = list(range(10))
        Xs, ys = bootstrap_sample(X, y)
        assert len(Xs) == 10 and len(ys) == 10
        logging.info("test_bootstrap passed.")
    except Exception as e:
        logging.error("test_bootstrap failed.", exc_info=True)
        raise e

def test_split() -> None:
    """
    Test the get_split function for data partitioning.
    """
    try:
        dataset = [[2, 0], [3, 0], [10, 1], [11, 1]]
        node = get_split(dataset)
        assert 'index' in node
        logging.info("test_split passed.")
    except Exception as e:
        logging.error("test_split failed.", exc_info=True)
        raise e

def test_evaluate() -> None:
    """
    Test the evaluate_model function for correct confusion matrix and report generation.
    """
    try:
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 1, 1]
        cm, report = evaluate_model(y_true, y_pred)
        assert cm.shape == (2, 2)
        logging.info("test_evaluate passed.")
    except Exception as e:
        logging.error("test_evaluate failed.", exc_info=True)
        raise e

def run_all_tests() -> None:
    """
    Run all test cases.
    """
    test_decision_tree()
    test_random_forest()
    test_gini()
    test_bootstrap()
    test_split()
    test_evaluate()
    logging.info("All tests completed successfully.")

if __name__ == "__main__":
    run_all_tests()

