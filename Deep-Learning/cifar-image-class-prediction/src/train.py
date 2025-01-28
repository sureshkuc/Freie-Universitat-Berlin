"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to train and validate PyTorch models.
Version: 1.0
"""

import torch
import logging
from statistics import mean

def train(model, optimizer, scheduler, data, max_epochs, cuda=True):
    """
    Train a PyTorch model with the provided data, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        data (DataLoader or tuple): Training data loader or a tuple of (train_loader, test_loader).
        max_epochs (int): Maximum number of training epochs.
        cuda (bool, optional): Whether to use GPU for training. Defaults to True.

    Returns:
        dict: The state dictionary of the best model (based on validation loss).
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting training")

    use_test = isinstance(data, tuple) and len(data) == 2
    if use_test:
        train_loader, test_loader = data
    else:
        train_loader = data

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    best_model = None
    min_loss = float('inf')

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")
        epoch_loss = []
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        accuracy = correct / total
        mean_loss = mean(epoch_loss)
        logger.info(f"Train Loss: {mean_loss:.4f}, Train Accuracy: {accuracy:.2%}")

        if use_test:
            test_loss, test_accuracy = validate(model, test_loader, criterion, cuda)
            if test_loss < min_loss:
                min_loss = test_loss
                best_model = model.state_dict()

            logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}")

        scheduler.step()

    return best_model

def validate(model, data_loader, criterion, cuda=True):
    """
    Validate a PyTorch model on the provided data loader.

    Args:
        model (torch.nn.Module): The model to validate.
        data_loader (DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        cuda (bool, optional): Whether to use GPU for validation. Defaults to True.

    Returns:
        tuple: Mean validation loss and accuracy.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting validation")

    model.eval()
    total_loss = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss.append(loss.item())
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

    mean_loss = mean(total_loss)
    accuracy = correct / total
    logger.info(f"Validation Loss: {mean_loss:.4f}, Validation Accuracy: {accuracy:.2%}")
    return mean_loss, accuracy

if __name__ == "__main__":
    """
    Entry point for the script. Initializes logging and provides an example usage.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Train script initialized. Define your model, optimizer, scheduler, and data to begin training.")

