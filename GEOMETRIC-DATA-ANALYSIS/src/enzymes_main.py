"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes for graph classification using Graph Convolutional Networks (GCN). 
    The `GraphClassifier` class implements a model to classify graph data, with data preprocessing, model training,
    testing, accuracy plotting, and k-fold cross-validation functionality.
Version: 1.0
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GCNConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Setup logging
log_dir = "outputs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class GraphClassifier:
    """
    A class used to represent a Graph Classifier based on Graph Convolutional Networks (GCN).

    Attributes
    ----------
    dataset : torch_geometric.data.Dataset
        The dataset to be used for training and testing the classifier.
    batch_size : int, optional
        The batch size used during training (default is 32).
    lr : float, optional
        The learning rate for the optimizer (default is 0.0005).
    device : torch.device
        The device on which the model will be run (either 'cpu' or 'cuda').
    model : nn.Module
        The Graph Neural Network model.
    optimizer : torch.optim.Adam
        The optimizer used for training.
    train_loader : DataLoader
        DataLoader for the training dataset.
    test_loader : DataLoader
        DataLoader for the test dataset.

    Methods
    -------
    prepare_data():
        Prepares the training and testing data.
    train(epoch):
        Trains the model for one epoch.
    test(loader):
        Tests the model on a given data loader.
    plot_acc(epochs, train_losses, test_losses):
        Plots the training and testing accuracy over epochs.
    k_fold_validation(k=10):
        Performs k-fold cross-validation on the dataset.
    """

    def __init__(self, dataset, batch_size=32, lr=0.0005):
        """
        Initializes the GraphClassifier with the provided dataset and hyperparameters.

        Parameters
        ----------
        dataset : torch_geometric.data.Dataset
            The dataset to be used for training and testing the classifier.
        batch_size : int, optional
            The batch size used during training (default is 32).
        lr : float, optional
            The learning rate for the optimizer (default is 0.0005).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loader, self.test_loader = self.prepare_data()

    def prepare_data(self):
        """
        Prepares the training and testing data by splitting the dataset.

        Returns
        -------
        tuple
            A tuple containing the training and testing DataLoader objects.
        """
        try:
            dataset = self.dataset.shuffle()
            n = len(dataset) // 10
            test_dataset = dataset[:n]
            train_dataset = dataset[n:]
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
            return train_loader, test_loader
        except Exception as e:
            logging.error(f"Error in data preparation: {e}")
            raise

    class Net(nn.Module):
        """
        A simple Graph Neural Network model for graph classification using GCNConv and TopKPooling layers.

        Methods
        -------
        forward(data):
            Defines the forward pass of the model.
        """

        def __init__(self):
            """
            Initializes the layers of the graph neural network.
            """
            super().__init__()
            self.conv1 = GCNConv(dataset.num_features, 128)
            self.pool1 = TopKPooling(128, ratio=0.8)
            self.conv2 = GCNConv(128, 128)
            self.pool2 = TopKPooling(128, ratio=0.8)
            self.conv3 = GCNConv(128, 128)
            self.pool3 = TopKPooling(128, ratio=0.8)
            self.lin_skip1 = nn.Linear(dataset.num_features, 128)
            self.lin_skip2 = nn.Linear(128, 128)
            self.lin_skip3 = nn.Linear(128, 128)
            self.lin1 = nn.Linear(256, 128)
            self.lin2 = nn.Linear(128, 64)
            self.lin3 = nn.Linear(64, dataset.num_classes)

        def forward(self, data):
            """
            Defines the forward pass of the model, including graph convolutions and pooling operations.

            Parameters
            ----------
            data : torch_geometric.data.Data
                The input data object containing node features, edge indices, and graph batch information.

            Returns
            -------
            torch.Tensor
                The predicted class scores for each graph.
            """
            try:
                x, edge_index, batch = data.x, data.edge_index, data.batch

                x = F.relu(self.conv1(x, edge_index) + self.lin_skip1(x))
                x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
                x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

                x = F.relu(self.conv2(x, edge_index) + self.lin_skip2(x))
                x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
                x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

                x = F.relu(self.conv3(x, edge_index) + self.lin_skip3(x))
                x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
                x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

                x = x1 + x2 + x3

                x = F.relu(self.lin1(x))
                x = F.dropout(x, p=0.5, training=self.training)
                x = F.relu(self.lin2(x))
                x = F.log_softmax(self.lin3(x), dim=-1)

                return x
            except Exception as e:
                logging.error(f"Error in forward pass: {e}")
                raise

    def train(self, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Returns
        -------
        float
            The average loss for the epoch.
        """
        try:
            self.model.train()
            loss_all = 0
            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, data.y)
                loss.backward()
                loss_all += data.num_graphs * loss.item()
                self.optimizer.step()
            return loss_all / len(self.dataset)
        except Exception as e:
            logging.error(f"Error during training at epoch {epoch}: {e}")
            raise

    def test(self, loader):
        """
        Tests the model on a given data loader.

        Parameters
        ----------
        loader : DataLoader
            The data loader for the dataset to be tested.

        Returns
        -------
        float
            The accuracy of the model on the given data.
        """
        try:
            self.model.eval()
            correct = 0
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data).max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
            return correct / len(loader.dataset)
        except Exception as e:
            logging.error(f"Error during testing: {e}")
            raise

    def plot_acc(self, epochs, train_losses, test_losses):
        """
        Plots the training and testing accuracy over epochs.

        Parameters
        ----------
        epochs : int
            The number of epochs for the plot.
        train_losses : list
            The list of training loss values.
        test_losses : list
            The list of testing loss values.
        """
        try:
            plt.rcParams['figure.figsize'] = [10, 5]
            plt.rcParams['figure.dpi'] = 100
            plt.plot(range(epochs), train_losses, label='train acc')
            plt.plot(range(epochs), test_losses, label='test acc')
            plt.title('Sparse Hierarchical Graph Classifier')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"Error during accuracy plot: {e}")
            raise

    def k_fold_validation(self, k=10):
        """
        Performs k-fold cross-validation on the dataset.

        Parameters
        ----------
        k : int, optional
            The number of folds for cross-validation (default is 10).
        """
        try:
            kfold = KFold(n_splits=k, shuffle=True)
            for fold, (train_idx, test_idx) in enumerate(kfold.split(self.dataset)):
                logging.info(f"Fold {fold + 1}/{k}")
                train_data = self.dataset[train_idx]
                test_data = self.dataset[test_idx]
                self.train_loader = DataLoader(train_data, batch_size=self.batch_size)
                self.test_loader = DataLoader(test_data, batch_size=self.batch_size)
                
                for epoch in range(1, 101):
                    loss = self.train(epoch)
                    test_acc = self.test(self.test_loader)
                    logging.info(f"Epoch {epoch}: Train Loss {loss:.4f}, Test Accuracy {test_acc:.4f}")

        except Exception as e:
            logging.error(f"Error during k-fold validation: {e}")
            raise

