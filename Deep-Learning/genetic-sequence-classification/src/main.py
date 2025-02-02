"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to load and preprocess biological sequence data,
    define and train RNN and LSTM models, and evaluate their performance.
Version: 1.0
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from config import *
from train import train
from rnn import RNNModel
from lstm import LSTMModel
from evaluation import plot_loss, plot_acc
import numpy as np


def char_to_int(data: list[str], vocab: list[str]) -> np.ndarray:
    """
    Converts a list of character sequences into integer sequences.

    Args:
        data (list[str]): A list of sequences, where each sequence is a string of characters.
        vocab (list[str]): A list of unique characters representing the vocabulary.

    Returns:
        np.ndarray: A 2D array where each character is replaced by its integer index.
    """
    char2idx = {u: i for i, u in enumerate(vocab)}
    text_as_int = [np.array([char2idx[c] for c in text]) for text in data]
    return np.array(text_as_int)


def one_hot_encode(arr: np.ndarray, n_labels: int = 4, max_sequence_len: int = 2000) -> np.ndarray:
    """
    One-hot encodes a list of integer sequences.

    Args:
        arr (np.ndarray): A 2D array where each entry is an integer representing a character.
        n_labels (int): The number of unique labels (default is 4 for DNA sequences A, C, G, T).
        max_sequence_len (int): The maximum sequence length to pad sequences to (default is 2000).

    Returns:
        np.ndarray: A 3D array of one-hot encoded sequences with padding applied.
    """
    encoded = []
    eye = np.eye(n_labels)
    for code in arr:
        encoded.append(np.vstack([
            np.array([eye[c] for c in code]),
            np.zeros((max_sequence_len - len(code), n_labels))
        ]))
    return np.array(encoded).swapaxes(1, 2)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the data from the .npz file.

    Returns:
        tuple: A tuple containing the training, validation, and test data arrays:
            (train_x, train_y, val_x, val_y, test_x)
    """
    with np.load('rnn-challenge-data.npz') as f:
        train_x = f['data_x']
        train_y = f['data_y']
        val_x = f['val_x']
        val_y = f['val_y']
        test_x = f['test_x']
    return train_x, train_y, val_x, val_y, test_x


def prepare_dataloaders() -> tuple[DataLoader, DataLoader]:
    """
    Prepares the training and testing DataLoaders.

    Returns:
        tuple: A tuple containing the training and testing DataLoaders.
    """
    train_x, train_y, val_x, val_y, test_x = load_data()
    
    # Data preprocessing steps
    text_as_int = char_to_int(train_x, vocab)
    X_train = one_hot_encode(text_as_int)
    y_train = np.eye(5)[train_y]

    text_as_int = char_to_int(val_x, vocab)
    X_val = one_hot_encode(text_as_int)
    y_val = np.eye(5)[val_y]

    text_as_int = char_to_int(test_x, vocab)
    X_test = one_hot_encode(text_as_int)

    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train_and_evaluate(model, optimizer, train_loader: DataLoader, test_loader: DataLoader) -> None:
    """
    Trains and evaluates the model.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_loader (DataLoader): The DataLoader for training data.
        test_loader (DataLoader): The DataLoader for test data.
    """
    train_losses, test_losses, train_acc, test_acc = train(model, optimizer, (train_loader, test_loader), max_epochs=MAX_EPOCHS)

    plot_loss(MAX_EPOCHS, train_losses, test_losses, model.__class__.__name__)
    plot_acc(MAX_EPOCHS, train_acc, test_acc, model.__class__.__name__)


def main() -> None:
    """
    Main function to load data, initialize models, and perform training and evaluation.
    """
    train_loader, test_loader = prepare_dataloaders()

    # Initialize and train models
    rnn_model = RNNModel(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    rnn_optimizer = get_optimizer(rnn_model)
    train_and_evaluate(rnn_model, rnn_optimizer, train_loader, test_loader)

    lstm_model = LSTMModel(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_dim=HIDDEN_DIM, n_layers=1)
    lstm_optimizer = get_optimizer(lstm_model)
    train_and_evaluate(lstm_model, lstm_optimizer, train_loader, test_loader)


if __name__ == "__main__":
    main()

