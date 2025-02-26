"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a class and functions to perform sparse graph classification using Graph Convolutional Networks (GCN)
    and Top-K pooling layers. It supports training, testing, and cross-validation with K-folds. Results are logged and errors 
    are handled gracefully. The module also includes model training and evaluation along with result visualization.
Version: 1.0
"""

import os
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
import numpy as np
import time
from sklearn.model_selection import KFold
from typing import Dict, Any

# Set up logging
log_folder = 'outputs'
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, f'log_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log')

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class SparseGraphClassifier:
    """
    Sparse Graph Classifier using Graph Convolutional Networks (GCN) and Top-K pooling.

    Attributes:
        dataset_name (str): The name of the dataset to use for classification.
        batch_size (int): The batch size for training and testing.
        epochs (int): The number of epochs for training.
        k_folds (int): The number of K-folds for cross-validation.
        device (torch.device): The device on which the model will run (CPU or GPU).
        dataset (TUDataset): The dataset used for training and testing.
        test_loader (DataLoader): DataLoader for the test dataset.
        train_loader (DataLoader): DataLoader for the training dataset.
        model (torch.nn.Module): The model used for classification.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
    """

    def __init__(self, dataset_name: str = 'DD', batch_size: int = 32, epochs: int = 20, k_folds: int = 10) -> None:
        """
        Initializes the SparseGraphClassifier with specified parameters.

        Args:
            dataset_name (str): The name of the dataset to use for classification.
            batch_size (int): The batch size for training and testing.
            epochs (int): The number of epochs for training.
            k_folds (int): The number of K-folds for cross-validation.
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_folds = k_folds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset
        self.dataset = TUDataset(root='/tmp/DD', name=self.dataset_name)
        self.dataset = self.dataset.shuffle()
        self.dataset_size = len(self.dataset)

        # Split dataset
        self.n = self.dataset_size // 10
        self.test_dataset = self.dataset[:self.n]
        self.train_dataset = self.dataset[self.n:]

        # Initialize DataLoader
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)

        # Initialize model and optimizer
        self.model = self.Net(self.dataset.num_features, self.dataset.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

    class Net(torch.nn.Module):
        """
        Graph Convolutional Network (GCN) model with Top-K pooling and residual connections.

        Args:
            num_features (int): The number of input features for each node.
            num_classes (int): The number of output classes.
        """

        def __init__(self, num_features: int, num_classes: int) -> None:
            super().__init__()
            self.conv1 = GCNConv(num_features, 64)
            self.pool1 = TopKPooling(64, ratio=0.8)
            self.conv2 = GCNConv(64, 64)
            self.pool2 = TopKPooling(64, ratio=0.8)
            self.conv3 = GCNConv(64, 64)
            self.pool3 = TopKPooling(64, ratio=0.8)
            self.lin_skip1 = torch.nn.Linear(num_features, 64)
            self.lin_skip2 = torch.nn.Linear(64, 64)
            self.lin_skip3 = torch.nn.Linear(64, 64)
            self.lin1 = torch.nn.Linear(128, 128)
            self.lin2 = torch.nn.Linear(128, 64)
            self.lin3 = torch.nn.Linear(64, num_classes)

        def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
            """
            Forward pass of the model.

            Args:
                data (torch_geometric.data.Data): The input graph data.

            Returns:
                torch.Tensor: The model's output.
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index) + self.lin_skip1(x))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            x1 = torch.cat([pyg_nn.global_max_pool(x, batch), pyg_nn.global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index) + self.lin_skip2(x))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x2 = torch.cat([pyg_nn.global_max_pool(x, batch), pyg_nn.global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index) + self.lin_skip3(x))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x3 = torch.cat([pyg_nn.global_max_pool(x, batch), pyg_nn.global_mean_pool(x, batch)], dim=1)

            x = x1 + x2 + x3
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.lin2(x))
            x = F.log_softmax(self.lin3(x), dim=-1)

            return x

    def train(self, epoch: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        loss_all = 0
        for data in self.train_loader:
            try:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, data.y)
                loss.backward()
                loss_all += data.num_graphs * loss.item()
                self.optimizer.step()
            except Exception as e:
                logging.error(f"Error during training epoch {epoch}: {e}")
        return loss_all / len(self.train_dataset)

    def test(self, loader: DataLoader) -> float:
        """
        Evaluates the model on a given dataset loader.

        Args:
            loader (DataLoader): The DataLoader for the dataset to evaluate.

        Returns:
            float: The accuracy of the model on the dataset.
        """
        self.model.eval()
        correct = 0
        for data in loader:
            try:
                data = data.to(self.device)
                pred = self.model(data).max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
            except Exception as e:
                logging.error(f"Error during testing: {e}")
        return correct / len(loader.dataset)

    def cross_validate(self) -> Dict[str, Any]:
        """
        Performs K-fold cross-validation on the dataset.

        Returns:
            dict: A dictionary containing the performance history of each fold.
        """
        splits = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        foldperf = {}
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(self.dataset)))):
            try:
                logging.info(f"Starting fold {fold + 1}")
                history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
                test_dataset = self.dataset[val_idx]
                train_dataset = self.dataset[train_idx]
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

                for epoch in range(1, self.epochs + 1):
                    loss = self.train(epoch)
                    train_acc = self.test(train_loader)
                    test_acc = self.test(test_loader)
                    history['train_loss'].append(loss)
                    history['train_acc'].append(train_acc)
                    history['test_acc'].append(test_acc)
                    logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, Test Acc: {test_acc:.5f}')

                foldperf[f'fold_{fold + 1}'] = history
            except Exception as e:
                logging.error(f"Error during cross-validation fold {fold + 1}: {e}")
        return foldperf

# Example usage
if __name__ == "__main__":
    try:
        model = SparseGraphClassifier()
        fold_performance = model.cross_validate()
        logging.info(f"Cross-validation results: {fold_performance}")
    except Exception as e:
        logging.error(f"Error in model execution: {e}")

