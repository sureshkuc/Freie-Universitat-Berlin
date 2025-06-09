"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module prepares the Setosa dataset, trains two perceptron models
    (a custom single-layer perceptron and a multilayer perceptron),
    evaluates their performance, and logs all events and errors.
Version: 1.0
"""

import os
import logging
from typing import Tuple, Any

from utils import prepare_setosa_data
from train_myperceptron import train_myperceptron_model
from train_multilayerperceptron import train_multilayerperceptron_model
from evaluation import evaluate_model


# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/model_pipeline.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_model_pipeline(
    model_trainer: Any,
    model_name: str
) -> None:
    """
    Executes the training and evaluation pipeline for a given model.

    Args:
        model_trainer (Callable): A function that trains the model and returns a fitted model.
        model_name (str): The name to be used in logging and evaluation.

    Raises:
        Exception: If any part of the pipeline fails, logs the error.
    """
    try:
        logging.info("Starting pipeline for model: %s", model_name)

        X_train, X_test, y_train, y_test = prepare_setosa_data()
        logging.debug("Data preparation complete for model: %s", model_name)

        model = model_trainer(X_train, y_train)
        logging.info("Model training completed: %s", model_name)

        evaluate_model(model, X_test, y_test, model_name=model_name)
        logging.info("Model evaluation completed: %s", model_name)

    except Exception as e:
        logging.exception("Error occurred during the pipeline for model: %s", model_name)
        raise e


def main() -> None:
    """Main function to run pipelines for both perceptron models."""
    logging.info("=== Starting model training pipelines ===")

    run_model_pipeline(train_myperceptron_model, model_name="MyPerceptron")
    run_model_pipeline(train_multilayerperceptron_model, model_name="MultilayerPerceptron")

    logging.info("=== All pipelines completed successfully ===")


if __name__ == "__main__":
    main()

