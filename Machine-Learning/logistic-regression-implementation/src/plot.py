"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions to visualize sample images and prediction 
    distributions for machine learning tasks. It includes functionality to 
    display random sample images from the dataset and to plot the distribution 
    of prediction probabilities.
Version: 1.0
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

# Ensure the 'outputs' directory exists
os.makedirs("outputs", exist_ok=True)

# Set up logging
logging.basicConfig(
    filename='outputs/plot.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

def show_sample_images(X: Union[np.ndarray, list]) -> None:
    """
    Display a grid of randomly selected sample images.

    Args:
        X (Union[np.ndarray, list]): A collection of images, each expected to be
        reshaped into a 16x16 grayscale image.

    Raises:
        ValueError: If the input array is not large enough to sample 90 images.
    """
    try:
        if len(X) < 90:
            raise ValueError("The input array must contain at least 90 images.")

        indices = np.random.choice(len(X), 90, replace=False)
        samples = np.array(X)[indices]

        fig = plt.figure(figsize=(20, 6))
        for i, img in enumerate(samples):
            ax = plt.subplot(6, 15, i + 1)
            plt.imshow(1 - np.reshape(img, (16, 16)), cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        logging.info("Displayed 90 sample images successfully.")

    except Exception as e:
        logging.error("Error in show_sample_images: %s", str(e), exc_info=True)


def plot_prediction_distribution(predictions: Union[np.ndarray, list]) -> None:
    """
    Plot a histogram showing the distribution of prediction probabilities.

    Args:
        predictions (Union[np.ndarray, list]): A sequence of prediction probabilities.

    Raises:
        ValueError: If predictions is empty or contains non-numeric values.
    """
    try:
        if len(predictions) == 0:
            raise ValueError("The predictions array is empty.")

        predictions = np.array(predictions)
        if not np.issubdtype(predictions.dtype, np.number):
            raise ValueError("Predictions must be numeric.")

        plt.hist(predictions, bins=10, color='skyblue', edgecolor='black')
        plt.title('Prediction Distribution')
        plt.xlabel('Probabilities')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        logging.info("Displayed prediction distribution histogram successfully.")

    except Exception as e:
        logging.error("Error in plot_prediction_distribution: %s", str(e), exc_info=True)

