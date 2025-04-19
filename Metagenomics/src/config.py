"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes for model configuration, optimizer and scheduler setup,
    and logging functionality. It includes a basic setup for training parameters and error handling.
Version: 1.0
"""

import os
import logging
import torch
import torch.optim as optim

# Logging configuration
LOG_PATH = 'outputs'

# Ensure the log folder exists
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_PATH, 'training.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Model hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 100
CUDA_AVAILABLE = torch.cuda.is_available()

def get_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Returns the optimizer for the given model.

    Args:
        model (torch.nn.Module): The PyTorch model to optimize.

    Returns:
        torch.optim.Optimizer: The optimizer configured for the model.
    """
    try:
        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
        logging.info("Optimizer created successfully.")
        return optimizer
    except Exception as e:
        logging.error(f"Error in get_optimizer: {e}")
        raise

def get_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.MultiStepLR:
    """
    Returns the learning rate scheduler for the given optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create a scheduler.

    Returns:
        torch.optim.lr_scheduler.MultiStepLR: The learning rate scheduler.
    """
    try:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 200], gamma=0.1
        )
        logging.info("Scheduler created successfully.")
        return scheduler
    except Exception as e:
        logging.error(f"Error in get_scheduler: {e}")
        raise

