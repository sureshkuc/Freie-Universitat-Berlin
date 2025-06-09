"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides an implementation of a Multilayer Perceptron (MLP)
    for supervised learning tasks using customizable activation functions
    and stochastic gradient descent for training.

Version: 1.0
"""

import os
import logging
import numpy as np
from config import LEARNING_RATE
from typing import List, Tuple, Dict, Callable

# Ensure log directory exists
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/mlp.log",
    level=logging.DEBUG,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

__all__ = ['ActivationFunctions', 'MultilayerPerceptron']


class ActivationFunctions:
    """Namespace for common activation functions and their gradients."""

    class Sigmoid:
        """Sigmoid activation function."""

        @staticmethod
        def activation(z: np.ndarray) -> np.ndarray:
            """Apply sigmoid activation."""
            return 1 / (1 + np.exp(-z))

        @staticmethod
        def gradient(z: np.ndarray) -> np.ndarray:
            """Compute the gradient of the sigmoid function."""
            act = ActivationFunctions.Sigmoid.activation(z)
            return act * (1 - act)

    class Relu:
        """ReLU activation function."""

        @staticmethod
        def activation(z: np.ndarray) -> np.ndarray:
            """Apply ReLU activation."""
            return np.maximum(0, z)

        @staticmethod
        def gradient(z: np.ndarray) -> np.ndarray:
            """Compute the gradient of the ReLU function."""
            return np.where(z > 0, 1, 0)

    class Softmax:
        """Softmax activation function (usually for output layer)."""

        @staticmethod
        def activation(z: np.ndarray) -> np.ndarray:
            """Apply softmax activation."""
            exps = np.exp(z - np.max(z))
            return exps / np.sum(exps)


class MultilayerPerceptron:
    """
    Multilayer Perceptron (MLP) neural network for classification tasks.
    
    Attributes:
        depth (int): Number of layers in the network.
        learning_rate (float): Learning rate for weight updates.
        sizes (List[int]): Number of neurons per layer.
        w (Dict[int, np.ndarray]): Weights for each layer.
        b (Dict[int, np.ndarray]): Biases for each layer.
        activations (Dict[int, Callable]): Activation functions for each layer.
    """

    def __init__(
        self,
        dimensions: List[int],
        activations: List[Callable],
        learning_rate: float = LEARNING_RATE
    ) -> None:
        """
        Initialize the MLP with layer dimensions and activation functions.

        Args:
            dimensions: List of integers specifying layer sizes.
            activations: List of activation function classes.
            learning_rate: Learning rate for gradient descent.
        """
        self.depth = len(dimensions)
        self.learning_rate = learning_rate
        self.sizes = dimensions
        self.w: Dict[int, np.ndarray] = {}
        self.b: Dict[int, np.ndarray] = {}
        self.activations: Dict[int, Callable] = {}

        try:
            for i in range(len(dimensions) - 1):
                self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
                self.b[i + 1] = np.zeros((1, dimensions[i + 1]))
                self.activations[i + 2] = activations[i]
        except Exception as e:
            logging.exception("Error during MultilayerPerceptron initialization.")
            raise

    def _feed_forward(self, x: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Perform the forward pass through the network.

        Args:
            x: Input feature vector.

        Returns:
            A tuple of dictionaries: (z values, activation values).
        """
        z: Dict[int, np.ndarray] = {}
        a: Dict[int, np.ndarray] = {1: x.reshape(1, -1)}

        try:
            for i in range(1, self.depth - 1):
                z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
                a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            return z, a
        except Exception as e:
            logging.exception("Error during feed-forward pass.")
            raise

    def grad(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Compute gradients of loss with respect to weights and biases.

        Args:
            x: Input feature vector.
            y: One-hot encoded target vector.

        Returns:
            Tuple of gradients (dW, dB).
        """
        try:
            Z, A = self._feed_forward(x)
            dW: Dict[int, np.ndarray] = {}
            dB: Dict[int, np.ndarray] = {}
            dZ: Dict[int, np.ndarray] = {}

            L = self.depth - 1
            dZ[L] = A[L] - y

            for k in range(L, 1, -1):
                dW[k - 1] = np.dot(A[k - 1].T, dZ[k])
                dB[k - 1] = np.sum(dZ[k], axis=0, keepdims=True)
                dA = np.dot(dZ[k], self.w[k].T)
                dZ[k - 1] = dA * self.activations[k].gradient(Z[k])
            return dW, dB
        except Exception as e:
            logging.exception("Error computing gradients.")
            raise

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int) -> None:
        """
        Train the MLP on the given dataset.

        Args:
            X: Feature matrix.
            Y: One-hot encoded labels.
            epochs: Number of training iterations.
        """
        try:
            for _ in range(epochs):
                for x, y in zip(X, Y):
                    dW, dB = self.grad(x, y)
                    for k in range(1, self.depth):
                        self.w[k] -= self.learning_rate * dW[k - 1]
                        self.b[k] -= self.learning_rate * dB[k - 1]
        except Exception as e:
            logging.exception("Error during MLP training.")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            X: Input feature matrix.

        Returns:
            Predicted class indices.
        """
        try:
            predictions = []
            for x in X:
                _, a = self._feed_forward(x)
                predictions.append(np.argmax(a[self.depth - 1]))
            return np.array(predictions)
        except Exception as e:
            logging.exception("Error during prediction.")
            raise

