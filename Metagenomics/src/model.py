"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to define and build a Convolutional Neural Network (CNN) model.
    It includes the model architecture and a helper function to build the model with error handling and logging.
Version: 1.0
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import logging

# Setup logging to log all error levels in 'outputs' folder
logging.basicConfig(filename='outputs/model.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Flatten(nn.Module):
    """
    A custom layer to flatten the input tensor.
    
    Args:
        nn.Module: Inherited class from PyTorch to define custom layers.
    
    Methods:
        forward(x): Flattens the input tensor to a 2D tensor.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flattens the input tensor to a 2D tensor.
        
        Args:
            x (torch.Tensor): The input tensor to be flattened.
        
        Returns:
            torch.Tensor: The flattened tensor.
        """
        return x.view(x.size(0), -1)


class CNNModel(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for processing 1D input data.
    
    Args:
        nn.Module: Inherited class from PyTorch to define the CNN model.
    
    Methods:
        forward(x): Defines the forward pass of the CNN model.
    """
    
    def __init__(self):
        """
        Initializes the CNN model with multiple convolutional layers,
        batch normalization, pooling layers, and fully connected layers.
        """
        super(CNNModel, self).__init__()
        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=1)),
            ('relu1', nn.LeakyReLU()),
            ('bn1', nn.BatchNorm1d(5)),
            ('pool1', nn.MaxPool1d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5, stride=1, padding=1)),
            ('relu2', nn.LeakyReLU()),
            ('bn2', nn.BatchNorm1d(10)),
            ('pool2', nn.MaxPool1d(kernel_size=2, stride=2)),
            ('flatten', Flatten()),
            ('linearlayer1', nn.Linear(2540, 500)),
            ('relu1', nn.LeakyReLU()),
            ('dropout', nn.Dropout(p=0.5)),
            ('linearlayer2', nn.Linear(500, 96)),
            ('softmax', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN model.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        b, features = x.shape
        x = x.reshape([b, 1, features])
        return self.body(x)


def build_model() -> CNNModel:
    """
    Builds the CNN model and handles any errors during the construction.
    
    Returns:
        CNNModel: The CNN model instance.
    
    Raises:
        Exception: If there is an error during model construction.
    """
    try:
        model = CNNModel()
        logging.info("Model built successfully.")
        return model
    except Exception as e:
        logging.error(f"Error while building model: {str(e)}")
        raise


