"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to load, preprocess, and prepare 
    data for training a neural network model, specifically using a ResNet. 
    The module includes functionality for data loading, preprocessing, and setting up 
    the DataLoader for training and validation.
Version: 1.0
"""

import torch
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from config import DATA_PATH, OUTPUT_PREDICTION, BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, EPOCHS, CUDA_AVAILABLE, logger
from resnet import ResNet
from train import train
import torchvision.transforms as transforms


def preprocess(image):
    """
    Preprocesses a given image by normalizing and transposing the channels to match the format
    expected by the model.

    Args:
        image (numpy.ndarray): Input image to be preprocessed.

    Returns:
        numpy.ndarray: Preprocessed image with normalized pixel values and channel order adjusted.
    """
    image = np.array(image)
    cifar_mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, -1)
    cifar_std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, -1)
    image = (image - cifar_mean) / cifar_std
    return image.transpose(2, 0, 1)  # Change channel order to (C, H, W)


def main():
	"""
	Main function to load data, preprocess it, split it into training and validation sets, 
	and prepare the DataLoader. It also initializes the ResNet model and starts the training process.
	"""
	# Load data
	logger.info("Loading data")
	with np.load(DATA_PATH) as fh:
	    x_train = fh['x_train']
	    y_train = fh['y_train']
	    x_test = fh['x_test']

	logger.info(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
	logger.info(f"Test data shape: {x_test.shape}")

	# Data preprocessing
	logger.info("Preprocessing data")
	x_train = np.array([preprocess(img) for img in x_train], dtype=np.float32)
	x_test = np.array([preprocess(img) for img in x_test], dtype=np.float32)

	# Split data into training and validation sets
	X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

	# Create DataLoader
	train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
	val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

	# Define ResNet model
	logger.info("Initializing ResNet model")
	model = ResNet()

	# Train the model
	logger.info("Starting training process")
	train(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, CUDA_AVAILABLE)

if __name__ == "__main__":
    main()

