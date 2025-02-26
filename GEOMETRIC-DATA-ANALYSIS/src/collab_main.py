"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides a graph classification framework using graph neural networks (GNNs) implemented with PyTorch Geometric.
    It includes a GraphClassifier class for loading datasets, training, testing, and performing k-fold cross-validation.
    The model architecture consists of Graph Convolutional Networks (GCNs) with TopKPooling layers for graph-level classification tasks.
Version: 1.0
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from datetime import datetime
from sklearn.model_selection import KFold
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling, global_max_pool as gmp, global_mean_pool as gap


class GraphClassifier:
    """
    A class to handle graph classification tasks using a Graph Neural Network (GNN).

    Attributes:
        device (torch.device): The device to run the model on (CPU or GPU).
        batch_size (int): The batch size for training and testing.
        k_folds (int): The number of folds for cross-validation.
        dataset_name (str): The name of the dataset to be used.
        dataset (TUDataset): The loaded graph dataset.
        dataset_size (int): The size of the dataset.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Adam): The optimizer used for training.
    
    Methods:
        setup_logging: Initializes logging setup for tracking model training and errors.
        log_error: Logs error messages during execution.
        load_dataset: Loads and shuffles the specified dataset.
        setup_model: Initializes the model and optimizer.
        train: Trains the model for one epoch and returns the average loss.
        test: Evaluates the model on the given dataset and returns the accuracy.
        plot_acc: Plots training and testing accuracy over epochs.
        k_fold_validation: Performs k-fold cross-validation and tracks performance.
        summarize_results: Logs the summary of the k-fold cross-validation results.
    """
    
    def __init__(self, dataset_name: str, batch_size: int = 32, k_folds: int = 10) -> None:
        """
        Initializes the GraphClassifier with dataset and training parameters.

        Args:
            dataset_name (str): The name of the dataset to load.
            batch_size (int, optional): The batch size for training and testing. Defaults to 32.
            k_folds (int, optional): The number of folds for cross-validation. Defaults to 10.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.dataset_name = dataset_name
        self.setup_logging()
        self.load_dataset()
        self.setup_model()

    def setup_logging(self) -> None:
        """Sets up logging to track training progress and errors."""
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        logging.basicConfig(
            filename=os.path.join('outputs', f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def log_error(self, error_msg: str) -> None:
        """
        Logs an error message to the log file.

        Args:
            error_msg (str): The error message to log.
        """
        logging.error(error_msg)

    def load_dataset(self) -> None:
        """
        Loads and shuffles the specified dataset.
        Logs the dataset loading status.
        """
        try:
            self.dataset = TUDataset(root='/tmp/COLLAB', name=self.dataset_name).shuffle()
            self.dataset_size = len(self.dataset)
            logging.info(f'Dataset {self.dataset_name} loaded with {self.dataset_size} graphs.')
        except Exception as e:
            self.log_error(f"Error loading dataset: {str(e)}")
            raise

    def setup_model(self) -> None:
        """
        Sets up the model and optimizer. Logs the setup process.
        """
        try:
            self.model = Net(self.dataset).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
            logging.info("Model and optimizer set up successfully.")
        except Exception as e:
            self.log_error(f"Error setting up the model: {str(e)}")
            raise

    def train(self, epoch: int, train_loader: DataLoader) -> float:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            train_loader (DataLoader): The data loader for the training dataset.

        Returns:
            float: The average loss for the epoch.
        """
        self.model.train()
        loss_all = 0
        for data in train_loader:
            try:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, data.y)
                loss.backward()
                loss_all += data.num_graphs * loss.item()
                self.optimizer.step()
            except Exception as e:
                self.log_error(f"Error during training at epoch {epoch}: {str(e)}")
                raise
        return loss_all / len(train_loader.dataset)

    def test(self, loader: DataLoader) -> float:
        """
        Evaluates the model on the given dataset.

        Args:
            loader (DataLoader): The data loader for the dataset.

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
                self.log_error(f"Error during testing: {str(e)}")
                raise
        return correct / len(loader.dataset)

    def plot_acc(self, epochs: int, train_acc: list, test_acc: list) -> None:
        """
        Plots the training and testing accuracy over epochs.

        Args:
            epochs (int): The number of epochs to plot.
            train_acc (list): The training accuracy values.
            test_acc (list): The testing accuracy values.
        """
        try:
            plt.rcParams['figure.figsize'] = [10, 5]
            plt.rcParams['figure.dpi'] = 100
            plt.plot(range(epochs), train_acc, label='train acc')
            plt.plot(range(epochs), test_acc, label='test acc')
            plt.title('Sparse Hierarchical Graph Classifier')
            plt.legend()
            plt.show()
        except Exception as e:
            self.log_error(f"Error plotting accuracy: {str(e)}")
            raise

    def k_fold_validation(self) -> None:
        """
        Performs k-fold cross-validation and tracks the model's performance.
        Logs the results of each fold and summarizes them at the end.
        """
        splits = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_perf = {}

        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(self.dataset_size))):
            logging.info(f"Starting Fold {fold + 1}")
            history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

            train_dataset = self.dataset[train_idx]
            test_dataset = self.dataset[val_idx]
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

            for epoch in range(1, 30):
                try:
                    loss = self.train(epoch, train_loader)
                    train_acc = self.test(train_loader)
                    test_acc = self.test(test_loader)
                    logging.info(f"Epoch {epoch} - Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, Test Acc: {test_acc:.5f}")
                    history['train_loss'].append(loss)
                    history['train_acc'].append(train_acc)
                    history['test_acc'].append(test_acc)
                except Exception as e:
                    self.log_error(f"Error during fold {fold + 1}, epoch {epoch}: {str(e)}")
                    continue

            fold_perf[f'fold{fold + 1}'] = history

        self.summarize_results(fold_perf)

    def summarize_results(self, fold_perf: dict) -> None:
        """
        Summarizes and logs the results of the k-fold cross-validation.

        Args:
            fold_perf (dict): A dictionary containing performance metrics for each fold.
        """
        tl_f, ta_f, testa_f = [], [], []

        for f in range(1, self.k_folds + 1):
            tl_f.append(np.mean(fold_perf[f'fold{f}']['train_loss']))
            ta_f.append(np.mean(fold_perf[f'fold{f}']['train_acc']))
            testa_f.append(np.mean(fold_perf[f'fold{f}']['test_acc']))

        avg_train_loss = np.mean(tl_f)
        avg_train_acc = np.mean(ta_f)
        avg_test_acc = np.mean(testa_f)

        logging.info(f"Average Train Loss: {avg_train_loss:.5f}")
        logging.info(f"Average Train Accuracy: {avg_train_acc:.5f}")
        logging.info(f"Average Test Accuracy: {avg_test_acc:.5f}")

        self.plot_acc(30, ta_f, testa_f)


class Net(torch.nn.Module):
    """
    Defines the Graph Neural Network (GNN) for graph classification.

    The network consists of multiple GCN layers followed by a TopKPooling layer to downsample the graph.
    The final output is a graph-level classification using a fully connected layer.
    """

    def __init__(self, dataset) -> None:
        """
        Initializes the network with GCN and pooling layers.

        Args:
            dataset (TUDataset): The dataset to be used for training and testing.
        """
        super(Net, self).__init__()

        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes

        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.pool = TopKPooling(128, ratio=0.8)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, data) -> torch.Tensor:
        """
        Defines the forward pass for the model.

        Args:
            data (torch_geometric.data.Data): The input graph data.

        Returns:
            torch.Tensor: The output predictions for the graph classification task.
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool(x, edge_index, None, data.batch)
        x = gmp(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # Example usage
    dataset_name = 'PROTEINS'
    classifier = GraphClassifier(dataset_name=dataset_name)
    classifier.k_fold_validation()

