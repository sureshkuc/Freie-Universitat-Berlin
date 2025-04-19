"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to visualize training and testing performance 
    metrics such as loss and accuracy using matplotlib. It includes error handling 
    and logging functionality to ensure traceability and debugging ease.
Version: 1.0
"""

import os
import logging
from typing import List

import matplotlib.pyplot as plt

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/plot_metrics.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def plot_loss(epochs: int, train_losses: List[float], test_losses: List[float]) -> None:
    """
    Plots the training and testing loss over epochs.

    Args:
        epochs (int): Total number of epochs.
        train_losses (List[float]): List of training loss values.
        test_losses (List[float]): List of testing loss values.

    Raises:
        Exception: If any error occurs during plotting.
    """
    try:
        logging.debug("Starting loss plot...")
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.rcParams['figure.dpi'] = 100
        plt.plot(range(epochs), train_losses, label='Train Loss')
        plt.plot(range(epochs), test_losses, label='Test Loss')
        plt.title('CNN Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logging.info("Loss plot completed successfully.")
    except Exception as e:
        logging.error(f"Error while plotting loss: {e}", exc_info=True)
        raise


def plot_accuracy(epochs: int, train_acc: List[float], test_acc: List[float]) -> None:
    """
    Plots the training and testing accuracy over epochs.

    Args:
        epochs (int): Total number of epochs.
        train_acc (List[float]): List of training accuracy values.
        test_acc (List[float]): List of testing accuracy values.

    Raises:
        Exception: If any error occurs during plotting.
    """
    try:
        logging.debug("Starting accuracy plot...")
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.rcParams['figure.dpi'] = 100
        plt.plot(range(epochs), train_acc, label='Train Accuracy')
        plt.plot(range(epochs), test_acc, label='Test Accuracy')
        plt.title('CNN Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logging.info("Accuracy plot completed successfully.")
    except Exception as e:
        logging.error(f"Error while plotting accuracy: {e}", exc_info=True)
        raise

