"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to train and validate a neural network for image classification
    using the MNIST dataset. It loads training and test data, preprocesses them, builds a neural network model, 
    and performs training and validation. Additionally, it visualizes some predictions from the model.
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
import torchvision
import torch.nn.functional as F


def load_data():
    """
    Loads the training and test data for the model.
    Returns:
        - data_x (ndarray): Input training data.
        - data_y (ndarray): Output training data.
        - test_x (ndarray): Input test data.
        - test_y (ndarray): Output test data.
    """
    with np.load('prediction-challenge-01-data.npz') as fh:
        data_x = fh['data_x']
        data_y = fh['data_y']
        test_x = fh['test_x']
    
    test_y = np.load('prediction.npy')

    # Display dataset dimensions and types
    print(data_x.shape, data_x.dtype)
    print(data_y.shape, data_y.dtype)
    print(test_x.shape, test_x.dtype)

    return data_x, data_y, test_x, test_y


class LoadMNIST(torch.utils.data.dataset.Dataset):
    """
    Custom Dataset class for loading MNIST-like data.

    Args:
        data_x (ndarray): Input data (images).
        data_y (ndarray): Target labels (digits).
        transform (callable, optional): A function/transform to apply to the data.
    """

    def __init__(self, data_x, data_y, transform=None):
        self.data = data_x
        self.labels = data_y
        self.transform = transform

    def __getitem__(self, index):
        """
        Returns a single item (image and label) from the dataset.
        
        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: (image, label)
        """
        data = self.data[index]
        labels = self.labels[index]
        if self.transform:
            data = self.transform(data)
        return data, labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)


def preprocess_data(data_x, data_y):
    """
    Preprocess the data by normalizing it and splitting it into training and validation sets.

    Args:
        data_x (ndarray): Input data (images).
        data_y (ndarray): Output labels (digits).

    Returns:
        - train_loader (DataLoader): DataLoader for the training data.
        - validation_loader (DataLoader): DataLoader for the validation data.
    """
    data = LoadMNIST(data_x, data_y)

    validation_part = 0.2
    split = round(len(data) * validation_part)

    train_data, valid_data = torch.utils.data.random_split(data, [len(data) - split, split])

    # Calculate mean and std for normalization
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))
    batch = next(iter(train_loader))
    x, y = batch
    mean_t, std_t = torch.mean(x), torch.std(x)

    # Define transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(mean_t), std=(std_t))
    ])

    # Apply transforms
    train_data.dataset.transform = transform
    valid_data.dataset.transform = transform

    # DataLoader for training and validation
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, mean_t, std_t


class NeuralNetwork(nn.Module):
    """
    A simple neural network model with fully connected layers.

    Args:
        None
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=1024),
            nn.Sigmoid(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        """
        Forward pass for the neural network.

        Args:
            x (Tensor): Input data.

        Returns:
            Tensor: Output from the network.
        """
        b, c, d, e = x.shape
        x = x.reshape([b, 28 * 28])
        return self.body(x)


class Net(nn.Module):
    """
    A Convolutional Neural Network model for image classification.

    Args:
        None
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x (Tensor): Input data (images).

        Returns:
            Tensor: Output predictions.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(train_loader, model, loss_fn, optimizer, max_epochs):
    """
    Trains the model on the training data.

    Args:
        train_loader (DataLoader): The DataLoader containing training data.
        model (nn.Module): The neural network model to train.
        loss_fn (Loss function): The loss function to use during training.
        optimizer (Optimizer): The optimizer for updating model parameters.
        max_epochs (int): The number of epochs to train.

    Returns:
        list: The list of loss values during training.
    """
    model.train()
    losses = []
    batch_total = len(train_loader)

    for epoch in range(max_epochs):
        samples_total = 0
        samples_correct = 0
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            yhat = torch.argmax(output, dim=1)

            samples_total += len(y)
            samples_correct += torch.sum(yhat == y)
            losses.append(loss.item())
            if batch_idx % 50 == 0:
                acc = float(samples_correct) / float(samples_total)
                sys.stdout.write(f'\rEpoch: {epoch + 1}/{max_epochs} Step: {batch_idx}/{batch_total} Loss: {loss.item():.6f} Acc: {acc:.2%}')
    return losses


def validate(model, validation_loader, loss_fn):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The trained model.
        validation_loader (DataLoader): The DataLoader containing validation data.
        loss_fn (Loss function): The loss function for validation.

    Returns:
        tuple: Validation loss and accuracy.
    """
    model.eval()
    criterion = loss_fn
    losses = []
    correct_samples = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss = criterion(output, y)
            yhat = torch.argmax(output, dim=1)
            losses.append(loss.item())
            correct_samples += torch.sum(yhat == y)
            total_samples += len(y)

    mean_losses = np.mean(losses)
    acc = float(correct_samples) / float(total_samples)
    print(f'Validation complete! Validation loss: {mean_losses:.6f}, Validation accuracy: {acc:.2%}')

    return mean_losses, acc


def main():
    """
    Main function to load data, initialize model, and train and validate the model.
    """
    data_x, data_y, test_x, test_y = load_data()
    train_loader, validation_loader, mean_t, std_t = preprocess_data(data_x, data_y)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = Net().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Training and Validation
    train_loss = train(train_loader, model, loss_fn, optimizer, max_epochs=10)
    validate(model, validation_loader, loss_fn)

    # Test prediction
    input_test = (torch.from_numpy(test_x) - mean_t) / std_t
    prediction = model(input_test)
    predictions = torch.argmax(prediction, dim=1)

    # Visualize predictions
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 10, figure=fig)
    row = 0
    mc = 0
    for col in range(0, 20):
        if col == 10:
            row += 1
            mc = 10
        ax = fig.add_subplot(gs[row, col - mc])
        ind = np.random.randint(low=0, high=test_x.shape[0])
        ax.imshow(test_x[ind, 0])
        ax.set_title(test_y[ind])

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

