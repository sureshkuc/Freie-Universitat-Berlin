"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module initializes logging, trains a model, evaluates its performance,
    and generates plots based on predictions. It is designed as a robust pipeline
    with proper error handling and modular structure.
Version: 1.0
"""

import logging
import os
from typing import Tuple

from .utils import setup_logging
from .config import LOG_FILE
from .train import train_model
from .plot import plot_model_report
from .evaluation import evaluate_model


class ModelPipeline:
    """Encapsulates the machine learning pipeline operations."""

    def __init__(self) -> None:
        """Initializes the ModelPipeline and sets up logging."""
        self._setup_output_directory()
        setup_logging(LOG_FILE)
        self.logger = logging.getLogger(__name__)
        self.logger.info("ModelPipeline initialized.")

    def _setup_output_directory(self) -> None:
        """Ensures the outputs directory exists for logs and other artifacts."""
        os.makedirs("outputs", exist_ok=True)

    def run(self) -> None:
        """Executes the training, evaluation, and plotting pipeline."""
        try:
            self.logger.info("Starting model training.")
            y_test, y_pred = self._train()
            
            self.logger.info("Evaluating model.")
            evaluation_report = self._evaluate(y_test, y_pred)
            print(evaluation_report)

            self.logger.info("Plotting model report.")
            self._plot(y_test, y_pred)

        except Exception as e:
            self.logger.critical("Critical failure in main execution", exc_info=True)

    def _train(self) -> Tuple[list, list]:
        """
        Trains the model.

        Returns:
            Tuple[list, list]: True labels and predicted labels.
        """
        return train_model()

    def _evaluate(self, y_test: list, y_pred: list) -> str:
        """
        Evaluates the trained model.

        Args:
            y_test (list): True labels.
            y_pred (list): Predicted labels.

        Returns:
            str: Evaluation summary.
        """
        return evaluate_model(y_test, y_pred)

    def _plot(self, y_test: list, y_pred: list) -> None:
        """
        Plots the model report.

        Args:
            y_test (list): True labels.
            y_pred (list): Predicted labels.
        """
        plot_model_report(y_test, y_pred)


if __name__ == "__main__":
    pipeline = ModelPipeline()
    pipeline.run()

