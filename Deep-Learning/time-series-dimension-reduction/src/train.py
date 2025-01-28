"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions for training and validating a model using PyTorch.
    The functions handle the training loop, loss calculation, and evaluation of the model on test data.
Version: 1.0
"""

import torch
from torch import nn, optim

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, 
          criterion: nn.Module, device: torch.device) -> float:
    """
    Trains the model on the provided training data.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): The optimizer for the training process.
        criterion (nn.Module): The loss function to compute the error.
        device (torch.device): The device (CPU or GPU) to run the model on.

    Returns:
        float: The average training loss for the current epoch.
    """
    model.train()  # Set the model to training mode
    train_loss = 0
    for data, future_data in train_loader:
        data, future_data = data.to(device), future_data.to(device)  # Move data to the correct device
        optimizer.zero_grad()  # Zero the gradients
        output = model(data.float())  # Forward pass
        loss = criterion(output, future_data.float())  # Compute the loss
        loss.backward()  # Backward pass
        train_loss += loss.item()  # Accumulate loss
        optimizer.step()  # Update the model parameters

    avg_loss = train_loss / len(train_loader.dataset)  # Compute average loss
    return avg_loss

def validate(model: nn.Module, test_loader: torch.utils.data.DataLoader, criterion: nn.Module, 
             device: torch.device) -> float:
    """
    Validates the model on the provided test data.

    Args:
        model (nn.Module): The model to be validated.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        criterion (nn.Module): The loss function to compute the error.
        device (torch.device): The device (CPU or GPU) to run the model on.

    Returns:
        float: The average test loss.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for data, future_data in test_loader:
            data, future_data = data.to(device), future_data.to(device)  # Move data to device
            output = model(data.float())  # Forward pass
            test_loss += criterion(output, future_data.float()).item()  # Accumulate loss

    avg_loss = test_loss / len(test_loader.dataset)  # Compute average loss
    return avg_loss

if __name__ == '__main__':
    # The main entry point for the script could be added here.
    pass

