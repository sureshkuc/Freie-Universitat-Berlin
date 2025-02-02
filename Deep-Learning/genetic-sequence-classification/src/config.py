"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to configure model parameters
    for training, including setting the optimizer and device configuration for
    a deep learning model.
Version: 1.0
"""

import torch
import torch.optim as optim

# Configuration Constants
INPUT_SIZE = 2000
OUTPUT_SIZE = 5
HIDDEN_DIM = 40
N_LAYERS = 3
BATCH_SIZE = 16
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3

# Device configuration
CUDA = torch.cuda.is_available()

def get_optimizer(model: torch.nn.Module) -> optim.Optimizer:
    """
    Creates an Adam optimizer for the given model with a predefined learning rate.

    Args:
        model (torch.nn.Module): The model for which the optimizer is to be created.

    Returns:
        optim.Optimizer: An Adam optimizer configured for the model.
    
    Example:
        model = YourModel()
        optimizer = get_optimizer(model)
    """
    return optim.Adam(model.parameters(), lr=LEARNING_RATE)

