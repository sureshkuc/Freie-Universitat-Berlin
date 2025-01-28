"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides an implementation of an Autoencoder class using PyTorch.
    The Autoencoder consists of an encoder and a decoder network designed to compress and reconstruct input data.
Version: 1.0
"""

import torch
from torch import nn

class Autoencoder(nn.Module):
    """
    Autoencoder network with an encoder and decoder.

    This class implements a simple feedforward autoencoder. The encoder reduces the 
    input data to a lower-dimensional latent space representation, while the decoder 
    reconstructs the input data from the latent representation.

    Attributes:
    - encoder (nn.Sequential): The encoder part of the autoencoder.
    - decoder (nn.Sequential): The decoder part of the autoencoder.

    Methods:
    - encode(x): Encodes the input data `x` into the latent space.
    - decode(z): Decodes the latent representation `z` back to the original space.
    - forward(x): Encodes and then decodes the input data `x`, returning the reconstructed data.
    """

    def __init__(self, inp_size=3, lat_size=1, dropout=0.5):
        """
        Initializes the Autoencoder.

        Args:
        - inp_size (int): The size of the input data.
        - lat_size (int): The size of the latent (bottleneck) space.
        - dropout (float): The dropout rate to apply after each layer for regularization.
        """
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(inp_size, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, lat_size)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(lat_size, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, inp_size)
        )

    def encode(self, x):
        """
        Encodes the input data `x` into the latent space representation.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The latent representation of the input data.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decodes the latent representation `z` back to the original space.

        Args:
        - z (torch.Tensor): The latent representation.

        Returns:
        - torch.Tensor: The reconstructed data.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Performs a forward pass through the autoencoder.

        This method first encodes the input `x` into the latent space and then decodes it back 
        to the original input space to produce the reconstruction.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The reconstructed input data.
        """
        return self.decode(self.encode(x))

