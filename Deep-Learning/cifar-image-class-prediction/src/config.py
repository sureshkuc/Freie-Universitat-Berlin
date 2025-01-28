"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides configuration settings for paths, logging, and hyperparameters
    used throughout the project. It includes constants and logging setup for consistency.
Version: 1.0
"""

import os
import logging
from pathlib import Path

# Paths
DATA_PATH = Path("prediction-challenge-02-data.npz")
OUTPUT_PREDICTION = Path("prediction.npy")

# Logging Configuration
LOG_LEVEL = logging.DEBUG
"""
LOG_LEVEL specifies the severity level of logs to be displayed.
Options include DEBUG, INFO, WARNING, ERROR, and CRITICAL.
"""
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
"""
LOG_FORMAT defines the format of log messages.
- %(asctime)s: Timestamp of the log.
- %(levelname)s: Severity level.
- %(message)s: Log message content.
"""
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger()

# Hyperparameters
BATCH_SIZE = 128
"""
BATCH_SIZE defines the number of samples per batch for data loading.
"""
LEARNING_RATE = 0.1
"""
LEARNING_RATE specifies the step size for the optimizer during training.
"""
MOMENTUM = 0.9
"""
MOMENTUM enhances the optimizer by helping it accelerate in the right direction.
"""
WEIGHT_DECAY = 5e-4
"""
WEIGHT_DECAY adds regularization to the optimizer to prevent overfitting by penalizing large weights.
"""
EPOCHS = 40
"""
EPOCHS defines the total number of iterations over the training dataset during training.
"""
CUDA_AVAILABLE = torch.cuda.is_available()
"""
CUDA_AVAILABLE checks whether a CUDA-compatible GPU is available for acceleration.
"""

