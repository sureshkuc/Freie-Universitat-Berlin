"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes for graph classification using
    the PyTorch Geometric library. It includes model definition, training, 
    testing, cross-validation, and data preparation. Additionally, logging 
    functionality has been implemented to track training and evaluation progress.
Version: 1.0
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import time
from datetime import datetime
import numpy as np
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling, global_max_pool as gmp, global_mean_pool as gap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import logging


# Set up logging
log_dir = 'outputs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'model_training.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class GraphClassifier(nn.Module):
    """
    A class defining the graph classification model.

    This class defines a Graph Convolutional Network (GCN) with hierarchical pooling 
    layers for graph classification tasks. The forward method processes the graph data 
    through the network and outputs the class probabilities.

    Attributes:
        conv1, conv2, conv3: Graph convolution layers for feature extraction.
        pool1, pool2, pool3: Top-K pooling layers for hierarchical feature aggregation.
        lin_skip1, lin_skip2, lin_skip3: Linear skip connections to enhance learning.
        lin1, lin2, lin3: Fully connected layers for classification.

    Methods:
        forward(data): Defines the forward pass of the model.
    """
    
    def __init__(self, dataset):
        """
        Initialize the GraphClassifier model.

        Args:
            dataset (torch_geometric.data.Dataset): The dataset to define the model.
        """
        super(GraphClassifier, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = GCNConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = GCNConv(64, 64)
        self.pool3 = TopKPooling(64, ratio=0.8)

        self.lin_skip1 = nn.Linear(dataset.num_features, 64)
        self.lin_skip2 = nn.Linear(64, 64)
        self.lin_skip3 = nn.Linear(64, 64)
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        """
        Define the forward pass of the model.

        Args:
            data (torch_geometric.data.Data): The input graph data.

        Returns:
            torch.Tensor: The output predictions for the input graph.
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


class GraphTrainer:
    """
    A class responsible for training and testing the GraphClassifier model.

    This class handles data preparation, model training, and evaluation. It uses
    the Adam optimizer and evaluates the model on the training and test datasets.

    Attributes:
        device: The device (CPU or GPU) on which the model is running.
        model: The GraphClassifier model.
        optimizer: The optimizer used for training the model.
        dataset: The dataset for training and testing.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.

    Methods:
        prepare_data(): Prepares the training and testing datasets.
        train(epoch): Trains the model for one epoch.
        test(loader): Evaluates the model on the given data loader.
        run_training(epochs): Runs the full training process for the specified epochs.
    """
    
    def __init__(self, dataset):
        """
        Initialize the GraphTrainer.

        Args:
            dataset (torch_geometric.data.Dataset): The dataset used for training and testing.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphClassifier(dataset).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.dataset = dataset
        self.train_loader = None
        self.test_loader = None

    def prepare_data(self):
        """
        Prepare the training and testing datasets.

        The dataset is shuffled, and a portion is split into training and testing sets.
        """
        try:
            dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
            dataset = dataset.shuffle()
            n = len(dataset) // 10
            test_dataset = dataset[:n]
            train_dataset = dataset[n:]
            self.test_loader = DataLoader(test_dataset, batch_size=32)
            self.train_loader = DataLoader(train_dataset, batch_size=32)
        except Exception as e:
            logging.error(f"Error in preparing data: {e}")
            raise

    def train(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average loss over the training dataset.
        """
        self.model.train()

        loss_all = 0
        try:
            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, data.y)
                loss.backward()
                loss_all += data.num_graphs * loss.item()
                self.optimizer.step()
        except Exception as e:
            logging.error(f"Error during training epoch {epoch}: {e}")
            raise
        return loss_all / len(self.dataset)

    def test(self, loader):
        """
        Test the model on the given data loader.

        Args:
            loader (torch_geometric.data.DataLoader): The data loader for testing.

        Returns:
            float: The accuracy of the model on the test dataset.
        """
        self.model.eval()
        correct = 0
        try:
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data).max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
        except Exception as e:
            logging.error(f"Error during testing: {e}")
            raise
        return correct / len(loader.dataset)

    def run_training(self, epochs=40):
        """
        Run the full training process.

        Args:
            epochs (int): The number of epochs to train the model.
        """
        try:
            for epoch in range(1, epochs + 1):
                loss = self.train(epoch)
                train_acc = self.test(self.train_loader)
                test_acc = self.test(self.test_loader)
                logging.info(f"Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, Test Acc: {test_acc:.5f}")
        except Exception as e:
            logging.error(f"Error during training loop: {e}")
            raise


def plot_acc(train_acc_list, test_acc_list):
    """
    Plot the accuracy of the model during training.

    Args:
        train_acc_list (list): A list of training accuracies for each epoch.
        test_acc_list (list): A list of test accuracies for each epoch.
    """
    try:
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.rcParams['figure.dpi'] = 100
        plt.plot(range(len(train_acc_list)), train_acc_list, label='Train Acc')
        plt.plot(range(len(test_acc_list)), test_acc_list, label='Test Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Testing Accuracy')
        plt.legend()
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting accuracy: {e}")
        raise


if __name__ == '__main__':
    dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
    trainer = GraphTrainer(dataset)
    trainer.prepare_data()
    trainer.run_training(epochs=40)

