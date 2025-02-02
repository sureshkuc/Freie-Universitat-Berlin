"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides an implementation of an LSTM-based model for sequence processing. 
    The model is bidirectional and can be used for tasks such as classification or regression.
Version: 1.0
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A bidirectional LSTM model for sequence processing.

    This model consists of an LSTM layer followed by two fully connected layers.
    The LSTM is bidirectional, meaning it processes the sequence in both forward and backward directions.

    Args:
        input_size (int): The number of expected features in the input (i.e., number of input features per time step).
        output_size (int): The number of output classes or the output dimension.
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        n_layers (int): The number of layers in the LSTM.

    Attributes:
        hidden_dim (int): The number of features in the hidden state.
        n_layers (int): The number of layers in the LSTM.
        lstm (nn.LSTM): The LSTM layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.

    Methods:
        forward(x): Defines the forward pass of the model.

    """
    
    def __init__(self, input_size: int, output_size: int, hidden_dim: int, n_layers: int):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): The number of features in the input sequence.
            output_size (int): The number of output classes or features.
            hidden_dim (int): The number of features in the hidden state of the LSTM.
            n_layers (int): The number of layers in the LSTM.

        """
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        # First fully connected layer
        self.fc1 = nn.Linear(hidden_dim * 4 * 2, 75)
        # Second fully connected layer
        self.fc2 = nn.Linear(75, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: The output of the model, which has shape (batch_size, output_size).

        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_dim).requires_grad_()

        # Pass through the LSTM layer
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Flatten the output of the LSTM layer
        out = out.contiguous().view(batch_size, -1)

        # Pass through the fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)

        return out

