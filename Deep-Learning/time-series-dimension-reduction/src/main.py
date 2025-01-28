"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to train and evaluate an autoencoder model on time-series data.
    It loads data, preprocesses it, and trains an autoencoder to predict time series, then evaluates the model
    using clustering accuracy metrics.
Version: 1.0
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from config import Config
from autoencoder import Autoencoder
from train import train, validate
from evaluation import Evaluator

def main():
    """
    Main function to load data, preprocess it, train the autoencoder model,
    evaluate its performance, and visualize the training and validation losses.
    
    It also performs the clustering of the encoded features and prints the optimal cluster mapping
    and validation accuracy.
    """
    # Load the data
    with np.load(Config.DATA_PATH) as fh:
        data_x = fh['data_x']
        validation_x = fh['validation_x']
        validation_y = fh['validation_y']

    evaluator = Evaluator()

    # Preprocess the data
    train_X, _ = evaluator.mean_free(data_x[:-1])
    train_Y, _ = evaluator.mean_free(data_x[1:])
    test_X, _ = evaluator.mean_free(validation_x[:-1])
    test_Y, _ = evaluator.mean_free(validation_x[1:])

    # Whitening of the data
    white_X, _ = evaluator.whiten_data(train_X)
    white_Y, _ = evaluator.whiten_data(train_Y)
    white_test_X, _ = evaluator.whiten_data(test_X)
    white_test_Y, _ = evaluator.whiten_data(test_Y)

    # Convert to PyTorch tensors and create DataLoader objects
    train_data = TensorDataset(torch.tensor(white_X), torch.tensor(white_Y))
    test_data = TensorDataset(torch.tensor(white_test_X), torch.tensor(white_test_Y))

    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE)

    # Initialize the model, optimizer, and loss function
    model = Autoencoder(inp_size=white_X.shape[1], lat_size=1).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []

    # Training loop
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, Config.DEVICE)
        test_loss = validate(model, test_loader, criterion, Config.DEVICE)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Config.EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, Config.EPOCHS + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    # Perform clustering on the encoded features and calculate validation accuracy
    with torch.no_grad():
        latent = model.encode(torch.tensor(white_test_X).float()).numpy()
    predictions, _ = evaluator.predict_kmeans(latent, n_clusters=4)
    clusters, max_accuracy = evaluator.validation_accuracy(predictions, validation_y)

    # Print optimal cluster mapping and validation accuracy
    print(f"Optimal Cluster Mapping: {clusters}")
    print(f"Max Validation Accuracy: {max_accuracy:.4f}")

if __name__ == "__main__":
    main()

