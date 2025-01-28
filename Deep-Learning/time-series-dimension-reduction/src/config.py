"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a configuration class for managing hyperparameters and device settings 
    required for deep learning models, including data path, batch size, learning rate, number of 
    epochs, and device (CUDA or CPU).
Version: 1.0
"""

import numpy as np
import torch

class Config:
    """
    Config class for storing the configuration parameters for the deep learning model.
    
    Attributes:
        DATA_PATH (str): Path to the dataset file.
        BATCH_SIZE (int): Number of samples per batch.
        LEARNING_RATE (float): Learning rate for model training.
        EPOCHS (int): Number of epochs for model training.
        DEVICE (torch.device): The device to be used for model training, either 'cuda' or 'cpu'.
    """
    
    DATA_PATH = 'dimredux-challenge-01-data.npz'
    BATCH_SIZE = 100
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

