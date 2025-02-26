"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to train and evaluate KNN models using both a custom implementation
    and Scikit-Learn's KNeighborsClassifier. It includes logging for debugging and error handling.
Version: 1.0
"""

import os
import logging
from train import train_knn
from evaluation import evaluate_knn
from config import load_data
from sklearn.neighbors import KNeighborsClassifier
from typing import NoReturn

# Configure logging
LOG_DIR = "outputs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "knn_training.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def train_sklearn_knn() -> NoReturn:
    """Trains KNN using Scikit-Learn's implementation and logs accuracy."""
    try:
        X_train, y_train, X_test, y_test = load_data()
        
        if X_train is None or y_train is None or X_test is None or y_test is None:
            logging.error("Failed to load dataset: Data returned None")
            return

        for k in range(1, 4):
            logging.info(f"Training Scikit-Learn KNN with k={k}")
            sklearn_model = KNeighborsClassifier(n_neighbors=k)
            sklearn_model.fit(X_train, y_train)
            accuracy = sklearn_model.score(X_test, y_test)
            logging.info(f"Scikit-Learn KNN Accuracy for k={k}: {accuracy:.4f}")
            print(f"Scikit-Learn KNN Accuracy for k={k}: {accuracy:.4f}")
    except Exception as e:
        logging.exception("Error occurred during Scikit-Learn KNN training")

if __name__ == "__main__":
    try:
        logging.info("Starting KNN Training")
        knn_model, results = train_knn()
        evaluate_knn(knn_model)
        train_sklearn_knn()
        logging.info("KNN Training Completed Successfully")
    except Exception as e:
        logging.exception("Unexpected error in the main execution block")

