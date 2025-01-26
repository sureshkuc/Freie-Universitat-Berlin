"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to train a neural network on a dataset, including
    the preprocessing steps like normalization, one-hot encoding, and training using backpropagation.
    The neural network consists of multiple layers with different activation functions like ReLU, Sigmoid, 
    and Softmax. The module also provides functions for visualizing the results, including the weights and 
    incorrectly classified images.
Version: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import os
import sklearn
%matplotlib inline


def show_numbers(X, y):
    """
    Displays random sample images from the dataset with their labels.

    Parameters:
    X : numpy array
        The input data.
    y : numpy array
        The labels for the data.
    """
    num_samples = 233
    indices = np.random.choice(range(len(X)), num_samples)
    sample_digits = X[indices]
    s_y = y[indices]
    fig = plt.figure(figsize=(20, 20))

    for i in range(num_samples):
        ax = plt.subplot(16, 15, i + 1)
        img = 255 - sample_digits[i].reshape((28, 28))
        plt.imshow(img, cmap='gray')
        plt.title(s_y[i])
        plt.axis('off')
    fig.tight_layout()
    plt.show()



class Sigmoid:
    """
    Sigmoid activation function and its gradient.
    """
    @staticmethod
    def activation(z):
        """
        Sigmoid activation function.

        Parameters:
        z : numpy array
            The input to the activation function.

        Returns:
        numpy array : The output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        """
        Gradient of the sigmoid function.

        Parameters:
        z : numpy array
            The input to the activation function.

        Returns:
        numpy array : The gradient of the sigmoid function.
        """
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))

class Relu:
    """
    ReLU activation function and its gradient.
    """
    @staticmethod
    def activation(z):
        """
        ReLU activation function.

        Parameters:
        z : numpy array
            The input to the activation function.

        Returns:
        numpy array : The output after applying the ReLU function.
        """
        z[z < 0] = 0
        return z

    @staticmethod
    def gradient(x):
        """
        Gradient of the ReLU function.

        Parameters:
        x : numpy array
            The input to the gradient function.

        Returns:
        numpy array : The gradient of the ReLU function.
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

class Softmax:
    """
    Softmax activation function.
    """
    @staticmethod
    def activation(Z):
        """
        Softmax activation function.

        Parameters:
        Z : numpy array
            The input to the activation function.

        Returns:
        numpy array : The output after applying the softmax function.
        """
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

class MultilayerPerceptron:
    """
    A Multilayer Perceptron neural network with backpropagation for training.
    """
    def __init__(self, total_layer=2, dimensions=None, activations=None, learning_rate=0.1):
        """
        Initializes the neural network.

        Parameters:
        total_layer : int
            The number of layers including input, hidden, and output layers.
        dimensions : list
            The dimensions of each layer.
        activations : list
            A list of activation functions for each layer.
        learning_rate : float
            The learning rate for training.
        """
        self.n_layers = total_layer
        self.loss = None
        self.learning_rate = learning_rate
        self.sizes = dimensions

        # Initializing weights and biases
        self.w = {}
        self.b = {}

        self.activations = {}

        for i in range(len(dimensions) - 1):
            limit = 1 / np.sqrt(dimensions[i])
            self.w[i + 1] = np.random.uniform(-limit, limit, (dimensions[i + 1], dimensions[i]))
            self.b[i + 1] = np.zeros((dimensions[i + 1], 1))
            self.activations[i + 2] = activations[i]

    def _feed_forward(self, x):
        """
        Executes a forward pass through the network.

        Parameters:
        x : numpy array
            The input data.

        Returns:
        tuple : Contains z values and activations for each layer.
        """
        z = {}
        a = {1: x.T}

        for i in range(1, self.n_layers):
            z[i + 1] = np.dot(self.w[i], a[i]) + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        return z, a

    def backpropagation(self, x, y):
        """
        Executes the backpropagation algorithm.

        Parameters:
        x : numpy array
            The input data.
        y : numpy array
            The expected output.
        """
        self.Z, self.A = self._feed_forward(x)
        self.dW = {}
        self.dB = {}
        self.dZ = {}
        self.dA = {}
        L = self.n_layers
        self.dZ[L] = (self.A[L] - y.T)

        for k in range(L, 1, -1):
            delta = np.dot(self.w[k - 1].T, self.dZ[k])
            self.dW[k - 1] = np.dot(self.dZ[k], self.A[k - 1].T) * (1 / self.total_samples)
            self.dB[k - 1] = np.sum(self.dZ[k], axis=1, keepdims=True) * (1 / self.total_samples)

            if k > 2:
                self.dZ[k - 1] = delta * self.activations[k - 1].gradient(self.A[k - 1])

    def fit(self, X, Y, epochs=100, display_loss=True, test_data=None):
        """
        Trains the model using backpropagation.

        Parameters:
        X : numpy array
            The input training data.
        Y : numpy array
            The training labels.
        epochs : int
            The number of epochs for training.
        display_loss : bool
            Whether to display the loss at each epoch.
        test_data : tuple
            A tuple containing test input data and test labels for evaluation.
        """
        if display_loss:
            train_loss = {}
            test_loss = {}
            train_acc = {}
            test_acc = {}

        self.total_samples = X.shape[0]
        for epoch in range(epochs):
            dW = {}
            dB = {}
            for i in range(self.n_layers - 1):
                dW[i + 1] = np.zeros((self.sizes[i + 1], self.sizes[i]))
                dB[i + 1] = np.zeros((self.sizes[i + 1], 1))

            X, Y = self.shuffle_data(X, Y)
            self.backpropagation(X, Y)

            for i in range(self.n_layers - 1):
                dW[i + 1] += self.dW[i + 1]
                dB[i + 1] += self.dB[i + 1]

            for i in range(self.n_layers - 1):
                self.w[i + 1] -= self.learning_rate * dW[i + 1]
                self.b[i + 1] -= self.learning_rate * dB[i + 1]

            if display_loss:
                train_loss[epoch] = self.cross_entropy(X, Y)
                test_loss[epoch] = self.cross_entropy(test_data[0], test_data[1])
                train_acc[epoch] = self.accuracy(X, Y)
                test_acc[epoch] = self.accuracy(test_data[0], test_data[1])

                if epoch % 50 == 0:
                    print("epoch", epoch, 'Train loss:', train_loss[epoch], 'Test loss:', test_loss[epoch], ' Train Acc:', train_acc[epoch], ' Test Acc:', test_acc[epoch])

        if display_loss:
            self.plot_loss_accuracy(train_loss, test_loss, train_acc, test_acc)

    def accuracy(self, X, Y):
        """
        Computes the accuracy of the model on the given data.

        Parameters:
        X : numpy array
            The input data.
        Y : numpy array
            The expected output.

        Returns:
        float : The accuracy of the model.
        """
        Z, A = self._feed_forward(X)
        Y_pred = np.argmax(A[self.n_layers], axis=0)
        Y_true = np.argmax(Y, axis=1)
        return np.mean(Y_pred == Y_true)

    def cross_entropy(self, X, Y):
        """
        Computes the cross-entropy loss for the given data.

        Parameters:
        X : numpy array
            The input data.
        Y : numpy array
            The expected output.

        Returns:
        float : The cross-entropy loss.
        """
        Z, A = self._feed_forward(X)
        return -np.sum(Y.T * np.log(A[self.n_layers])) / X.shape[0]

    def shuffle_data(self, X, Y):
        """
        Shuffles the dataset.

        Parameters:
        X : numpy array
            The input data.
        Y : numpy array
            The expected output.

        Returns:
        tuple : The shuffled input and expected output data.
        """
        return shuffle(X, Y)
    
    def plot_loss_accuracy(self, train_loss, test_loss, train_acc, test_acc):
        """
        Plots the loss and accuracy over epochs.

        Parameters:
        train_loss : dict
            The training loss for each epoch.
        test_loss : dict
            The testing loss for each epoch.
        train_acc : dict
            The training accuracy for each epoch.
        test_acc : dict
            The testing accuracy for each epoch.
        """
        fig, axs = plt.subplots(2, figsize=(12, 10))

        axs[0].plot(list(train_loss.keys()), list(train_loss.values()), label="Train Loss")
        axs[0].plot(list(test_loss.keys()), list(test_loss.values()), label="Test Loss")
        axs[0].set_title('Loss over Epochs')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(list(train_acc.keys()), list(train_acc.values()), label="Train Accuracy")
        axs[1].plot(list(test_acc.keys()), list(test_acc.values()), label="Test Accuracy")
        axs[1].set_title('Accuracy over Epochs')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plt.show()

        




def load_data(file_name):
    """
    Loads the dataset from the 'data' folder in the repository.

    Returns:
    tuple : Contains the training data, training labels, and test data.
    """
    data_path = os.path.join(os.getcwd(), 'data', file_name)
    
    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The data file was not found at {data_path}")
    
    with np.load(data_path) as fh:
        x_train = fh['x_train']
        y_train = fh['y_train']
        x_test = fh['x_test']
    
    print(f"Loaded data from {data_path}")
    return x_train, y_train, x_test


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Preprocesses the data by applying normalization and one-hot encoding.

    Parameters:
    x_train : numpy array
        The training data.
    y_train : numpy array
        The training labels.
    x_test : numpy array
        The test data.
    y_test : numpy array
        The test labels.

    Returns:
    tuple : Processed training data, training labels (one-hot encoded), test data, and test labels (one-hot encoded).
    """
    # One-hot encoding for labels
    enc = OneHotEncoder()
    y_OH_train = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
    y_OH_test = enc.fit_transform(np.expand_dims(y_test, 1)).toarray()

    # Data normalization
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, y_OH_train, x_test, y_OH_test, scaler


def build_mlp(x_train, y_train):
    """
    Builds the Multilayer Perceptron (MLP) model with defined layers and activations.

    Parameters:
    x_train : numpy array
        The input training data.
    y_train : numpy array
        The input training labels.

    Returns:
    mlp : MultilayerPerceptron
        The MLP model object.
    """
    dimensions = (x_train.shape[1], 400, 400, len(np.unique(y_train)))
    activations_funct_list = (Relu, Sigmoid, Softmax)
    mlp = MultilayerPerceptron(total_layer=len(dimensions), dimensions=dimensions, activations=activations_funct_list, learning_rate=0.1)
    return mlp


def evaluate_model(mlp, x_train, y_train, x_test, y_test):
    """
    Evaluates the model's performance on training and test datasets.

    Parameters:
    mlp : MultilayerPerceptron
        The trained model object.
    x_train : numpy array
        The input training data.
    y_train : numpy array
        The input training labels.
    x_test : numpy array
        The input test data.
    y_test : numpy array
        The input test labels.

    Returns:
    None
    """
    Y_pred_train = mlp.predict(x_train)
    Y_pred_train = np.argmax(Y_pred_train.T, 1)

    Y_pred_test = mlp.predict(x_test)
    Y_pred_test = np.argmax(Y_pred_test.T, 1)

    accuracy_train = accuracy_score(Y_pred_train, y_train)
    accuracy_test = accuracy_score(Y_pred_test, y_test)

    print("Training accuracy:", round(accuracy_train, 2))
    print("Test accuracy:", round(accuracy_test, 2))
    print(sklearn.metrics.classification_report(y_test, Y_pred_test))


def visualize_incorrect_classifications(x_test, Y_pred_test, y_test, scaler):
    """
    Visualizes incorrectly classified images.

    Parameters:
    x_test : numpy array
        The input test data.
    Y_pred_test : numpy array
        The predicted labels for the test data.
    y_test : numpy array
        The actual labels for the test data.
    scaler : StandardScaler
        The scaler object used to normalize the data.

    Returns:
    None
    """
    incorrectly_classified_images = x_test[Y_pred_test != y_test].copy()
    incorrectly_classified_images = scaler.inverse_transform(incorrectly_classified_images)
    print('Incorrectly classified digits:', incorrectly_classified_images.shape)
    show_numbers(incorrectly_classified_images, Y_pred_test[Y_pred_test != y_test])


def show_numbers(X, y):
    """
    Displays random sample images from the dataset with their labels.

    Parameters:
    X : numpy array
        The input data.
    y : numpy array
        The labels for the data.
    """
    num_samples = 233
    indices = np.random.choice(range(len(X)), num_samples)
    sample_digits = X[indices]
    s_y = y[indices]
    fig = plt.figure(figsize=(20, 20))

    for i in range(num_samples):
        ax = plt.subplot(16, 15, i + 1)
        img = 255 - sample_digits[i].reshape((28, 28))
        plt.imshow(img, cmap='gray')
        plt.title(s_y[i])
        plt.axis('off')
    fig.tight_layout()
    plt.show()


def show_weights(X):
    """
    Displays weights as images.

    Parameters:
    X : numpy array
        The weights to be displayed as images.
    """
    num_samples = X.shape[0]
    fig = plt.figure(figsize=(20, 20))
    for i in range(num_samples):
        ax = plt.subplot(20, 20, i + 1)
        img = 255 - X[i].reshape((28, 28))
        plt.imshow(img, cmap='gray')
        plt.axis('off')


def main():
    """
    Main function to load data, preprocess, train the model, and evaluate its performance.
    """
    # Load the data
    x_train, y_train, x_test = load_data('prediction-challenge-02-data.npz')
    
    # Preprocess the data
    x_train, y_OH_train, x_test, y_OH_test, scaler = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Build and train the model
    mlp = build_mlp(x_train, y_train)
    mlp.fit(x_train, y_OH_train, epochs=3000, display_loss=True, test_data=(x_test, y_OH_test))
    
    # Evaluate the model
    evaluate_model(mlp, x_train, y_train, x_test, y_test)
    
    # Visualize incorrectly classified images
    visualize_incorrect_classifications(x_test, Y_pred_test, y_test, scaler)
    
    # Visualize weights
    scaled_weights = (mlp.w[1] - np.min(mlp.w[1])) / (np.max(mlp.w[1]) - np.min(mlp.w[1])) * 255
    show_weights(scaled_weights)

    # Make predictions on new data
    new_data = test_x.reshape(test_x.shape[0], 784)
    new_data = scaler.transform(new_data)
    Y_pred_test = mlp.predict(new_data)
    prediction = Y_pred_test
    assert prediction.ndim == 1
    assert prediction.shape[0] == 2000
    np.save('prediction.npy', prediction)

if __name__ == "__main__":
    main()



