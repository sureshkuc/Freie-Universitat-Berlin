# config.py
"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    configuration parameters
Version: 1.0
"""

import os

# Paths
DATA_PATH = os.path.join("data", "prediction-challenge-01-data.npz")
TEST_LABELS_PATH = os.path.join("data", "prediction.npy")

# Training Parameters
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10

# Logging Configuration
LOG_FILE = "training_log.log"
LOG_LEVEL = "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

