"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module loads the Iris dataset, performs binary classification using a 
    linear SVM, evaluates multiple models for different regularization parameters, 
    and visualizes the results including decision boundaries and confusion matrices.
Version: 1.0
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_STATE, C_VALUES
from train import train_model
from model import LinearSVM
from evaluation import evaluate_model, get_classification_report, get_confusion_matrix
from plot import plot_decision_boundary, plot_confusion_matrix

# Setup logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/main.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main execution function to train and evaluate a linear SVM classifier
    on the Iris dataset (binary classification: Setosa vs Non-Setosa).
    
    Loads data, splits into training/testing sets, trains multiple models 
    for different regularization values, evaluates performance, 
    and plots results.
    
    Raises:
        Exception: If any error occurs during the pipeline execution.
    """
    try:
        logger.info("Starting the main pipeline...")

        # Load dataset
        iris = load_iris()
        df = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        logger.debug("Dataset loaded successfully.")

        # Convert multi-class to binary classification
        df = df.replace({'target': {2: 1, 0: -1}})
        X = df.drop(['target'], axis=1).values
        y = df['target'].values
        logger.debug("Converted to binary classification.")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info("Data split into train and test sets.")

        # Model training and evaluation
        best_f1 = -1.0
        best_model = (None, None)

        for C in C_VALUES:
            w, b = train_model(X_train, y_train, C)
            y_pred = LinearSVM.classify(w, b, X_test)
            acc, f1 = evaluate_model(y_test, y_pred)
            logger.info(f"C={C}, Accuracy={acc:.4f}, F1={f1:.4f}")
            print(f"C={C}, Accuracy={acc:.4f}, F1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = (w, b)

        # Use best model
        w, b = best_model
        y_pred = LinearSVM.classify(w, b, X_test)
        report = get_classification_report(
            y_test, y_pred, target_names=['Setosa', 'non Setosa']
        )
        cm = get_confusion_matrix(y_test, y_pred)

        print(report)
        plot_confusion_matrix(cm, ['Setosa', 'non Setosa'])
        plot_decision_boundary(X_test, y_test, w, b)

        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.exception("Error occurred in main execution.")
        raise

if __name__ == "__main__":
    main()

