"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides an implementation of a Recurrent Neural Network (RNN) model
    using PyTorch. The model is designed for sequence-based tasks and includes
    an RNN layer followed by fully connected layers for output prediction.
Version: 1.0
"""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    A Recurrent Neural Network (RNN) model for sequence prediction.

    Args:
        input_size (int): The number of features in the input data.
        output_size (int): The number of output classes or predictions.
        hidden_dim (int): The dimensionality of the hidden state.
        n_layers (int): The number of RNN layers.

    Methods:
        forward(x):
            Defines the forward pass of the model.
        
        init_hidden(batch_size):
            Initializes the hidden state of the RNN.
    """

    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        """
        Initializes the RNN model by defining the layers.

        Args:
            input_size (int): The number of features in the input data.
            output_size (int): The number of output classes or predictions.
            hidden_dim (int): The dimensionality of the hidden state.
            n_layers (int): The number of RNN layers.
        """
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # Fully connected layers for output prediction
        self.fc1 = nn.Linear(hidden_dim * 4, 75)
        self.fc2 = nn.Linear(75, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the RNN model.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor: The model's output predictions.
        """
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # RNN forward pass
        out, hidden = self.rnn(x, hidden)

        # Flatten the output and pass through fully connected layers
        out = out.contiguous().view(batch_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state of the RNN.

        Args:
            batch_size (int): The batch size for the current input.

        Returns:
            Tensor: The initialized hidden state tensor.
        """
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim)


# Main function for testing (optional)
if __name__ == "__main__":
    # Example usage of the RNN model
    input_size = 10  # Example input feature size
    output_size = 2  # Example output size (e.g., binary classification)
    hidden_dim = 64  # Hidden layer size
    n_layers = 2  # Number of RNN layers

    # Instantiate the model
    model = RNNModel(input_size, output_size, hidden_dim, n_layers)

    # Example input tensor (batch_size, seq_len, input_size)
    x = torch.randn(32, 50, input_size)

    # Forward pass
    output = model(x)

    print(output.shape)  # Should print (32, output_size)

