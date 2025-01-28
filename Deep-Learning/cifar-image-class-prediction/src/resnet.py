"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to implement a ResNet architecture,
    including ResidualBlock and ResNet classes for deep learning models.
Version: 1.0
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """Defines a residual block used in ResNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the convolution. Default is 1.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        logger.debug(f"Initializing ResidualBlock with in_channels={in_channels}, out_channels={out_channels}, stride={stride}")

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Forward pass of the residual block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the residual block.
        """
        logger.debug("Forward pass through ResidualBlock")
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNet(nn.Module):
    """Defines the ResNet architecture.

    Args:
        num_classes (int): Number of output classes. Default is 3.
    """
    def __init__(self, num_classes=3):
        super(ResNet, self).__init__()
        logger.debug(f"Initializing ResNet with num_classes={num_classes}")

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.block1 = self._create_block(64, 64, stride=1)
        self.block2 = self._create_block(64, 128, stride=2)
        self.block3 = self._create_block(128, 256, stride=2)
        self.block4 = self._create_block(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _create_block(self, in_channels, out_channels, stride):
        """Creates a sequence of residual blocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first block in the sequence.

        Returns:
            Sequential: A sequence of residual blocks.
        """
        logger.debug(f"Creating block with in_channels={in_channels}, out_channels={out_channels}, stride={stride}")
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        """Forward pass of the ResNet model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the ResNet model.
        """
        logger.debug("Forward pass through ResNet")
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def main():
    """Main function to test the ResNet model."""
    logger.info("Testing ResNet model")
    model = ResNet(num_classes=10)
    sample_input = torch.randn(1, 3, 32, 32)
    output = model(sample_input)
    print(f"Model output shape: {output.shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()

