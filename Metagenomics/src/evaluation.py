"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a validation utility for evaluating a PyTorch model's
    performance on a given dataset using cross-entropy loss. It also logs
    information and errors to assist with debugging and monitoring.
Version: 1.0
"""

import os
import logging
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader


# Ensure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Configure logging to write all log levels to 'outputs/evaluation.log'
logging.basicConfig(
    filename="outputs/evaluation.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def validate(
    model: torch.nn.Module,
    data: DataLoader,
    cuda: bool = True
) -> Tuple[float, float]:
    """
    Validates a PyTorch model using a given DataLoader.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        data (DataLoader): A DataLoader providing the validation dataset.
        cuda (bool, optional): Whether to use CUDA for computations. Defaults to True.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and accuracy.

    Raises:
        Exception: If an error occurs during validation.
    """
    try:
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        correct_samples = 0
        total_samples = 0

        for x, y in data:
            if cuda and torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            output = model(x)
            loss = criterion(output, y)
            y_pred = torch.argmax(output, dim=1)

            losses.append(loss.item())
            correct_samples += torch.sum(y_pred == y).item()
            total_samples += y.size(0)

        mean_loss = np.mean(losses)
        accuracy = float(correct_samples) / float(total_samples)

        logging.info(
            "Validation complete. Validation loss: %.6f, Accuracy: %.2f%%",
            mean_loss,
            accuracy * 100
        )

        return mean_loss, accuracy

    except Exception as e:
        logging.exception("Error occurred during model validation.")
        raise

