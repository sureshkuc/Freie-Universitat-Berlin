"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions for training and evaluating a neural network model using PyTorch.
    The `train` function trains a model on a given dataset and evaluates it on an optional test set.
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Union, Tuple
from statistics import mean
import sys

def train(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    data: Union[DataLoader, Tuple[DataLoader]], 
    max_epochs: int, 
    cuda: bool = False
) -> Tuple[list, list, list, list]:
    """
    Trains a model for a specified number of epochs and evaluates it optionally on a test set.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (optim.Optimizer): The optimizer for training.
        data (Union[DataLoader, Tuple[DataLoader]]): A DataLoader for training data, or a tuple containing both 
            a training DataLoader and a test DataLoader.
        max_epochs (int): The number of epochs to train the model.
        cuda (bool, optional): Whether to use GPU for training. Defaults to False.

    Returns:
        Tuple[list, list, list, list]: A tuple containing:
            - A list of training losses for each epoch.
            - A list of test losses for each epoch (if a test set is provided).
            - A list of training accuracies for each epoch.
            - A list of test accuracies for each epoch (if a test set is provided).
    
    Raises:
        TypeError: If the provided data argument is not a DataLoader or a tuple of two DataLoaders.
        ValueError: If the tuple of data is not of length 2.

    """
    
    use_test = False
    
    # Handle input data: training data only or training + test data
    if isinstance(data, DataLoader):
        train_loader = data
    elif isinstance(data, tuple):
        if len(data) == 2:
            train_loader, test_loader = data
            if not isinstance(train_loader, DataLoader):
                raise TypeError(f'Expected 1st entry of type DataLoader, but got {type(train_loader)}!')
            if not isinstance(test_loader, DataLoader):
                raise TypeError(f'Expected 2nd entry of type DataLoader, but got {type(test_loader)}!')
            use_test = True
        else:
            raise ValueError(f'Expected tuple of length 2, but got {len(data)}!')

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Set model to training mode
    model.train()
    
    # Initialize lists to store training and test results
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    
    batch_total = len(train_loader)

    for epoch in range(max_epochs):
        samples_total = 0
        samples_correct = 0
        losses = []
        t_losses = []

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            if cuda:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            yhat = torch.argmax(output, dim=1)

            samples_total += len(y)
            samples_correct += torch.sum(yhat == y)
            losses.append(loss.item())

        # Calculate training accuracy and loss
        acc = float(samples_correct) / float(samples_total)
        train_acc.append(acc)
        train_losses.append(mean(losses))
        
        # If test data is provided, evaluate on the test set
        if use_test:
            model.eval()
            samples_total = 0
            samples_correct = 0
            for test_x, test_y in test_loader:
                if cuda:
                    test_x, test_y = test_x.cuda(), test_y.cuda()
                test_output = model(test_x)
                test_yhat = torch.argmax(test_output, dim=1)
                loss = criterion(test_output, test_y)
                samples_total += len(test_y)
                samples_correct += torch.sum(test_yhat == test_y)
                t_losses.append(loss.item())
            
            t_acc = float(samples_correct) / float(samples_total)
            test_acc.append(t_acc)
            test_losses.append(mean(t_losses))
            model.train()

        # Print progress for each epoch
        sys.stdout.write(f'\rEpoch: {epoch}/{max_epochs} Loss: {mean(losses):.2f} Acc: {acc:.2%}')
    
    return train_losses, test_losses, train_acc, test_acc

