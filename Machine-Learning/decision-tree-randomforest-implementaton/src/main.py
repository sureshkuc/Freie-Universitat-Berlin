"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module orchestrates the execution of a machine learning pipeline.
    It reads data, trains a Decision Tree model, evaluates its performance, 
    and logs all events including errors to the 'outputs' folder.
Version: 1.0
"""

import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DATA_PATH,
    LOG_DIR,
    LOG_FILE,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
    RANDOM_STATE
)
from .model import DecisionTreeClassifier
from .evaluation import evaluate_model


def setup_logging(log_dir: str, log_file: str) -> None:
    """
    Sets up logging configuration to log messages at all severity levels.

    Args:
        log_dir (str): The directory where the log file will be stored.
        log_file (str): The complete path of the log file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # Also log to console


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads data from a CSV file and splits into features and labels.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y) arrays.
    """
    data = np.array(pd.read_csv(file_path, header=None))
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def run_pipeline() -> None:
    """
    Executes the end-to-end machine learning pipeline including:
    - Data loading
    - Splitting into training and test sets
    - Model training
    - Prediction
    - Evaluation
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info("Pipeline started.")

        X, y = load_data(DATA_PATH)
        logger.debug(f"Data loaded with shape: {X.shape}, Labels shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=RANDOM_STATE, stratify=y
        )
        logger.debug("Data split into training and test sets.")

        clf = DecisionTreeClassifier(
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF
        )
        clf.fit(X_train, y_train)
        logger.info("Model training completed.")

        y_pred = clf.predict(X_test)
        logger.debug("Prediction completed.")

        evaluate_model(y_test, y_pred)
        logger.info("Evaluation completed.")

    except Exception as e:
        logger.exception("An error occurred during pipeline execution.")


def main() -> None:
    """
    Main entry point of the script.
    """
    setup_logging(LOG_DIR, LOG_FILE)
    run_pipeline()


if __name__ == "__main__":
    main()

