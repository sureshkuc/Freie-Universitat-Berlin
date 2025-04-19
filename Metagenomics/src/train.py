"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a training utility function for PyTorch models,
    supporting optional evaluation using a validation/test dataloader.
    It includes detailed logging and error handling.
Version: 1.0
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Union, Tuple, Optional, List
from torch.utils.data import DataLoader
from statistics import mean

# Ensure the logging directory exists
os.makedirs('outputs', exist_ok=True)

# Setup logging
logging.basicConfig(
    filename='outputs/train.log',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    data: Union[DataLoader, Tuple[DataLoader, DataLoader]],
    max_epochs: int,
    cuda: bool = True
) -> Tuple[List[float], List[float], List[float], List[float], Optional[dict]]:
    """
    Trains a PyTorch model with optional validation.

    Args:
        model (nn.Module): The neural network model to train.
        optimizer (optim.Optimizer): Optimizer instance.
        scheduler (Optional[optim.lr_scheduler._LRScheduler]): Learning rate scheduler (can be None).
        data (Union[DataLoader, Tuple[DataLoader, DataLoader]]): Training dataloader or a tuple of (train_loader, test_loader).
        max_epochs (int): Number of training epochs.
        cuda (bool, optional): Whether to use CUDA for training. Defaults to True.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], Optional[dict]]: Training losses,
            test losses, training accuracies, test accuracies, and the best model weights (if test set used).
    """
    try:
        use_test = False
        if isinstance(data, DataLoader):
            train_loader = data
            test_loader = None
        elif isinstance(data, tuple) and len(data) == 2:
            train_loader, test_loader = data
            use_test = True
        else:
            raise ValueError(f'Expected DataLoader or tuple of length 2, but got {type(data)} with length {len(data) if isinstance(data, tuple) else "N/A"}.')

        criterion = nn.CrossEntropyLoss()
        model.train()

        losses, test_losses = [], []
        train_acc, test_acc = [], []
        best_model = None
        min_loss = float('inf')

        for epoch in range(max_epochs):
            samples_total = 0
            samples_correct = 0
            batch_loss = []

            for batch_idx, (x, y) in enumerate(train_loader):
                if cuda:
                    x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                yhat = torch.argmax(output, dim=1)
                samples_total += len(y)
                samples_correct += torch.sum(yhat == y).item()
                batch_loss.append(loss.item())

            acc = samples_correct / samples_total
            train_acc.append(acc)
            avg_train_loss = mean(batch_loss)
            losses.append(avg_train_loss)

            if scheduler is not None:
                scheduler.step()

            if use_test and test_loader is not None:
                model.eval()
                tsamples_total = 0
                tsamples_correct = 0
                test_epoch_loss = []

                with torch.no_grad():
                    for test_x, test_y in test_loader:
                        if cuda:
                            test_x, test_y = test_x.cuda(), test_y.cuda()

                        test_output = model(test_x)
                        test_loss = criterion(test_output, test_y)
                        test_epoch_loss.append(test_loss.item())

                        test_yhat = torch.argmax(test_output, dim=1)
                        tsamples_total += len(test_y)
                        tsamples_correct += torch.sum(test_yhat == test_y).item()

                tacc = tsamples_correct / tsamples_total
                test_acc.append(tacc)
                avg_test_loss = mean(test_epoch_loss)
                test_losses.append(avg_test_loss)

                if avg_test_loss < min_loss:
                    min_loss = avg_test_loss
                    best_model = model.state_dict()

                logging.info(
                    f"Epoch {epoch+1}/{max_epochs} | "
                    f"Train Loss: {avg_train_loss:.6f} | Train Acc: {acc:.2%} | "
                    f"Test Loss: {avg_test_loss:.6f} | Test Acc: {tacc:.2%}"
                )
            else:
                test_losses.append(0.0)  # placeholder
                logging.info(
                    f"Epoch {epoch+1}/{max_epochs} | "
                    f"Train Loss: {avg_train_loss:.6f} | Train Acc: {acc:.2%}"
                )

        logging.info("Training completed successfully.")
        return losses, test_losses, train_acc, test_acc, best_model

    except Exception as e:
        logging.exception("An error occurred during training.")
        raise

