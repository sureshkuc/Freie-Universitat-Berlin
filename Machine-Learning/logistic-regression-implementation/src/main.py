"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com

Description:
    This module loads training and testing data, filters and transforms labels,
    trains a logistic regression model, evaluates its performance, and visualizes
    prediction distributions and ROC curves.

Version: 1.0
"""

import os
import logging
from typing import NoReturn

import numpy as np
import pandas as pd

from config import TRAIN_PATH, TEST_PATH, setup_logging
from train import filter_digits, binary_labels, weight_initialization, find_optimum_lr
from model import LogisticRegression
from evaluation import evaluate_model, plot_roc_curve
from plot import show_sample_images, plot_prediction_distribution


def main() -> NoReturn:
    """
    Main pipeline function to load data, train the model, evaluate, and visualize results.

    This function performs the following steps:
        1. Loads training and testing data from CSV files.
        2. Filters digit classes and converts labels to binary.
        3. Initializes model weights and biases.
        4. Determines the optimal learning rate.
        5. Trains a logistic regression model.
        6. Evaluates model performance.
        7. Visualizes predictions and ROC curve.

    Raises:
        Exception: Catches and logs any unexpected exceptions during execution.
    """
    setup_logging(log_dir="outputs")  # Ensure logs are stored in outputs/

    try:
        # Load datasets
        train_data = np.array(pd.read_csv(TRAIN_PATH, sep=' ', header=None))
        test_data = np.array(pd.read_csv(TEST_PATH, sep=' ', header=None))

        X_train, y_train = train_data[:, 1:-1], train_data[:, 0]
        X_test, y_test = test_data[:, 1:], test_data[:, 0]

        # Filter digits and binarize labels
        X_train, y_train = filter_digits(X_train, y_train)
        X_test, y_test = filter_digits(X_test, y_test)

        y_train = binary_labels(y_train)
        y_test = binary_labels(y_test)

        # Display sample test images
        show_sample_images(X_test)

        # Initialize weights
        w, b = weight_initialization(X_train)
        num_iter = 500
        lr_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]

        # Find optimal learning rate
        optimum_lr = find_optimum_lr(
            lr_list, X_train, y_train, X_test, y_test, w, b, num_iter
        )

        # Train logistic regression model
        model = LogisticRegression(w, b).fit(X_train, y_train, optimum_lr, num_iter)

        # Predict and evaluate
        predictions = model.predict(X_test)
        score, report = evaluate_model(y_test, predictions)

        print(f"F1 Score: {score:.2f}")
        print(report)

        # Plot prediction analysis
        plot_prediction_distribution(predictions)
        plot_roc_curve(y_test, predictions)

        # Optionally train with MSE loss
        w, b = weight_initialization(X_train)
        _ = LogisticRegression(w, b).fit_mse(X_train, y_train, 0.05, 100)

    except Exception as e:
        logging.exception("An error occurred in the main pipeline")


if __name__ == '__main__':
    main()

