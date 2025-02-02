"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to plot the loss and accuracy curves 
    during model training and evaluation. It includes two functions: 
    `plot_loss` to visualize the loss curves and `plot_acc` to visualize 
    the accuracy curves.
Version: 1.0
"""

import matplotlib.pyplot as plt
from typing import List

def plot_loss(epochs: int, train_losses: List[float], test_losses: List[float], model_name: str) -> None:
    """
    Plots the training and testing loss over epochs.

    Args:
        epochs (int): The number of training epochs.
        train_losses (List[float]): A list of training loss values over epochs.
        test_losses (List[float]): A list of testing loss values over epochs.
        model_name (str): The name of the model to be displayed in the title.

    Returns:
        None: This function does not return any value, it just displays the plot.
    
    Example:
        plot_loss(100, train_losses, test_losses, "MyModel")
    """
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['figure.dpi'] = 100
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), test_losses, label='Test Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_acc(epochs: int, train_acc: List[float], test_acc: List[float], model_name: str) -> None:
    """
    Plots the training and testing accuracy over epochs.

    Args:
        epochs (int): The number of training epochs.
        train_acc (List[float]): A list of training accuracy values over epochs.
        test_acc (List[float]): A list of testing accuracy values over epochs.
        model_name (str): The name of the model to be displayed in the title.

    Returns:
        None: This function does not return any value, it just displays the plot.
    
    Example:
        plot_acc(100, train_acc, test_acc, "MyModel")
    """
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['figure.dpi'] = 100
    plt.plot(range(epochs), train_acc, label='Train Accuracy')
    plt.plot(range(epochs), test_acc, label='Test Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

