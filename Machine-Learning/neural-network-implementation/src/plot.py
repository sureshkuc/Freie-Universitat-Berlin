"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides visualization utilities for displaying sample images 
    from a dataset. It includes a class to visualize randomly selected images 
    in a grid format using Matplotlib.

Version: 1.0
"""

import os
import logging
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Set up logging
logging.basicConfig(
    filename="outputs/plot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SampleVisualizer:
    """
    A utility class for visualizing image samples from a dataset.
    """

    def __init__(self, sample_shape: Union[tuple[int, int], list[int]]) -> None:
        """
        Initialize the SampleVisualizer.

        Args:
            sample_shape: The shape (height, width) of each image sample.
        """
        self.sample_shape = sample_shape
        logging.debug(f"Initialized SampleVisualizer with shape: {self.sample_shape}")

    def show_samples(self, X: np.ndarray, num_samples: int = 90) -> None:
        """
        Display randomly selected image samples in a grid.

        Args:
            X: A NumPy array of image data.
            num_samples: The number of samples to display (default is 90).

        Raises:
            ValueError: If the dataset has fewer samples than requested.
        """
        try:
            if len(X) < num_samples:
                raise ValueError("Not enough samples to display.")

            indices = np.random.choice(len(X), num_samples, replace=False)
            fig = plt.figure(figsize=(20, 6))
            logging.info(f"Displaying {num_samples} image samples.")

            for i, idx in enumerate(indices):
                ax = plt.subplot(6, 15, i + 1)
                plt.imshow(X[idx].reshape(self.sample_shape), cmap="gray")
                plt.axis("off")

            plt.tight_layout()
            plt.show()
            logging.info("Sample visualization completed successfully.")

        except Exception as e:
            logging.error("An error occurred in show_samples()", exc_info=True)
            raise

# Example usage (commented for module cleanliness):
# if __name__ == "__main__":
#     data = np.random.rand(100, 256)  # Example dummy data with 16x16 images
#     visualizer = SampleVisualizer((16, 16))
#     visualizer.show_samples(data)

