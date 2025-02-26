"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to load dataset files, process training and testing data, 
    and log any errors encountered during the execution.
Version: 1.0
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Create 'outputs' folder if not exists
LOG_DIR = "outputs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
LOG_FILE = os.path.join(LOG_DIR, "app.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads training and testing data from CSV files.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        X_train (features), y_train (labels), X_test (features), y_test (labels).

    Raises:
        FileNotFoundError: If the data files are not found.
        ValueError: If the data files cannot be parsed correctly.
    """
    try:
        logging.info("Loading training and test data...")

        # Read training and test data
        training_data = np.array(pd.read_csv("Data/zip.train", sep=" ", header=None))
        test_data = np.array(pd.read_csv("Data/zip.test", sep=" ", header=None))

        # Extract features and labels
        X_train, y_train = training_data[:, 1:], training_data[:, 0]
        X_test, y_test = test_data[:, 1:], test_data[:, 0]

        logging.info("Data successfully loaded.")
        return X_train, y_train, X_test, y_test

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logging.error(f"Error parsing CSV files: {e}")
        raise


def main() -> None:
    """
    Main function to execute the data loading process.
    """
    try:
        X_train, y_train, X_test, y_test = load_data()
        logging.info("Data has been successfully loaded and is ready for processing.")

        # Example: Initialize a classifier (No training done yet)
        classifier = KNeighborsClassifier(n_neighbors=3)
        logging.info("KNeighborsClassifier initialized with k=3.")

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

