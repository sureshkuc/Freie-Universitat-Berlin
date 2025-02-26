"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to train a K-Nearest Neighbors (KNN) model using a dataset
    loaded from the configuration module. It includes logging and error handling for robustness.
Version: 1.0
"""

import os
import logging
from typing import Dict, Tuple
from model import KNearestNeighbors
from config import load_data

# Create output directory for logs
LOG_DIR = "outputs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
LOG_FILE = os.path.join(LOG_DIR, "training.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def train_knn() -> Tuple[KNearestNeighbors, Dict[int, float]]:
    """ 
    Trains the KNN model and evaluates its accuracy for different values of k.
    
    Returns:
        Tuple[KNearestNeighbors, Dict[int, float]]: The trained model and a dictionary of accuracies.
    """
    try:
        logging.info("Loading data...")
        X_train, y_train, X_test, y_test = load_data()
        
        logging.info("Initializing KNN model.")
        model = KNearestNeighbors()
        model.fit(X_train, y_train)

        results = {}
        for k in range(1, 4):
            logging.info(f"Predicting with k={k}")
            predictions = model.predict(X_test, k)
            accuracy = model.accuracy(y_test, predictions)
            results[k] = accuracy
            logging.info(f"Accuracy for k={k}: {accuracy:.4f}")
            print(f'Accuracy for k={k}: {accuracy:.4f}')
        
        return model, results
    
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    """Main execution block to train the KNN model."""
    try:
        trained_model, accuracies = train_knn()
        logging.info("Training completed successfully.")
    except Exception as err:
        logging.critical("Training failed due to an exception.", exc_info=True)

