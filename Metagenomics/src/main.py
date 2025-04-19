"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com

Description:
    This module provides functions and logic to load data, train a CNN model,
    evaluate its performance, and log the training process and results.

Version: 1.0
"""

import os
import time
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import CNNModel
from train import train
from evaluation import validate
from plot import plot_loss, plot_accuracy


# Configure logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/training.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BATCH_SIZE: int = 256
TEST_SIZE: float = 0.25
RANDOM_STATE: int = 1
LEARNING_RATE: float = 1e-2
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 5e-4
EPOCHS: int = 100


def load_data() -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Load and preprocess dataset for training and testing.

    Returns:
        Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]: Training loader,
        test loader, training features, and test features.
    """
    try:
        df = pd.read_csv("output_G.txt")
        y = np.array(df['Taxa_label'].astype('category').cat.codes)
        df = df.drop(columns=['seq_id', 'Taxa_label'])
        X = np.array(df.copy(), dtype=float)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            ),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            ),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        logging.info("Data loaded and split successfully.")
        return train_loader, test_loader, X_train, X_val

    except Exception as e:
        logging.error(f"Failed to load or preprocess data: {e}", exc_info=True)
        raise


def main() -> None:
    """
    Main function to train the CNN model, evaluate it, and plot metrics.
    """
    try:
        train_loader, test_loader, X_train, X_val = load_data()

        model = CNNModel()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 200], gamma=0.1
        )

        start_time = time.time()
        logging.info("Training started.")
        
        losses, test_losses, train_acc, test_acc, best_model = train(
            model, optimizer, scheduler, train_loader, test_loader,
            max_epochs=EPOCHS, cuda=torch.cuda.is_available()
        )

        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

        plot_loss(EPOCHS, losses, test_losses)
        plot_accuracy(EPOCHS, train_acc, test_acc)

        model.load_state_dict(best_model)
        validate(model, test_loader)

        logging.info("Model evaluation complete.")

    except Exception as e:
        logging.critical(f"Unhandled error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()

