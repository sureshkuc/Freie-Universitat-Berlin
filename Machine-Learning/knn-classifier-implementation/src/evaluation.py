"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to display digit images and evaluate the performance 
    of a K-Nearest Neighbors (KNN) model on test data. It includes proper logging and 
    error handling to track any issues encountered during execution.
Version: 1.0
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from config import load_data
from model import KNearestNeighbors
from typing import Any, Tuple

# Ensure output directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
LOG_FILE = os.path.join(OUTPUT_DIR, "app.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

def show_numbers(X: np.ndarray) -> None:
    """Displays a random set of digit images.
    
    Args:
        X (np.ndarray): The dataset containing digit images.
    
    Raises:
        ValueError: If X is empty or not a valid NumPy array.
    """
    try:
        if not isinstance(X, np.ndarray) or X.size == 0:
            raise ValueError("Input X must be a non-empty NumPy array.")
        
        num_samples = 90
        indices = np.random.choice(len(X), num_samples, replace=False)
        sample_digits = X[indices]

        fig = plt.figure(figsize=(20, 6))
        for i in range(num_samples):
            ax = plt.subplot(6, 15, i + 1)
            img = 255 - sample_digits[i].reshape((16, 16))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.show()
    except Exception as e:
        logging.error("Error in show_numbers: %s", str(e))

def evaluate_knn(model: Any) -> None:
    """Evaluates the KNN model on test data and displays misclassified images.
    
    Args:
        model (Any): An instance of the KNearestNeighbors classifier.
    
    Raises:
        RuntimeError: If data loading fails or prediction encounters an issue.
    """
    try:
        X_train, y_train, X_test, y_test = load_data()
        if X_test is None or y_test is None:
            raise RuntimeError("Failed to load test data.")

        predictions = model.predict(X_test, 1)
        misclassified_indices = np.where(predictions != y_test)[0]
        
        if misclassified_indices.size > 0:
            misclassified = X_test[misclassified_indices]
            print("Displaying misclassified digits...")
            show_numbers(misclassified)
        else:
            print("No misclassified digits found.")
    except Exception as e:
        logging.error("Error in evaluate_knn: %s", str(e))

def main() -> None:
    """Main execution function to run the KNN evaluation."""
    try:
        model = KNearestNeighbors()
        evaluate_knn(model)
    except Exception as e:
        logging.critical("Unhandled exception in main: %s", str(e))

if __name__ == "__main__":
    main()

