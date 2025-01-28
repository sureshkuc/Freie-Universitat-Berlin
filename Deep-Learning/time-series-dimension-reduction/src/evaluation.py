"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes for data evaluation, 
    including operations like mean subtraction, whitening, KMeans clustering, 
    accuracy calculation, and validation accuracy evaluation.
Version: 1.0
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

class Evaluator:
    """
    Evaluator class provides methods for performing data preprocessing, 
    clustering, and evaluating clustering performance.

    Methods:
    --------
    mean_free(data: np.ndarray) -> tuple:
        Subtracts the mean from the data.
    
    whiten_data(data: np.ndarray) -> tuple:
        Whitens the data using principal component analysis (PCA).

    predict_kmeans(latent: np.ndarray, n_clusters: int = 4) -> tuple:
        Performs KMeans clustering on the latent data.
    
    calculate_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
        Calculates the accuracy between true and predicted labels.

    validation_accuracy(my: np.ndarray, validation_y: np.ndarray) -> tuple:
        Finds the optimal mapping between predicted and true clusters to maximize accuracy.
    """

    @staticmethod
    def mean_free(data: np.ndarray) -> tuple:
        """
        Subtracts the mean from the data along each feature axis.

        Parameters:
        -----------
        data : np.ndarray
            The input data array to be mean-centered.

        Returns:
        --------
        tuple : (np.ndarray, np.ndarray)
            The mean-centered data and the mean vector.
        """
        means = data.sum(axis=0) / data.shape[0]
        return data - means, means

    @staticmethod
    def whiten_data(data: np.ndarray) -> tuple:
        """
        Whitens the data by removing correlations between features using PCA.

        Parameters:
        -----------
        data : np.ndarray
            The input data array to be whitened.

        Returns:
        --------
        tuple : (np.ndarray, np.ndarray)
            The whitened data and the covariance matrix used in whitening.
        """
        cov_matrix = np.cov(data.T)
        sigma, U = np.linalg.eig(cov_matrix)
        white_data = np.dot(data, U) @ np.diag(1 / np.sqrt(sigma))
        return white_data, cov_matrix

    @staticmethod
    def predict_kmeans(latent: np.ndarray, n_clusters: int = 4) -> tuple:
        """
        Performs KMeans clustering on the latent data.

        Parameters:
        -----------
        latent : np.ndarray
            The input data to be clustered.
        n_clusters : int, optional
            The number of clusters to form. Default is 4.

        Returns:
        --------
        tuple : (np.ndarray, KMeans)
            The predicted cluster labels and the KMeans model.
        """
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(latent)
        return kmeans.predict(latent), kmeans

    @staticmethod
    def calculate_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculates the accuracy between the true and predicted labels.

        Parameters:
        -----------
        true : np.ndarray
            The true labels.
        predicted : np.ndarray
            The predicted labels.

        Returns:
        --------
        float
            The accuracy score.
        """
        return accuracy_score(true, predicted)

    @staticmethod
    def validation_accuracy(my: np.ndarray, validation_y: np.ndarray) -> tuple:
        """
        Evaluates the accuracy by finding the optimal mapping of predicted clusters
        to true clusters to maximize the accuracy score.

        Parameters:
        -----------
        my : np.ndarray
            The predicted cluster labels.
        validation_y : np.ndarray
            The true labels.

        Returns:
        --------
        tuple : (np.ndarray, float)
            The optimal cluster mapping and the maximum accuracy score.
        """
        max_accuracy = accuracy_score(validation_y, my)
        clusters = np.array([0, 1, 2, 3])
        my = np.where(my == 0, 10, my)
        my = np.where(my == 1, 11, my)
        my = np.where(my == 2, 12, my)
        my = np.where(my == 3, 13, my)
        true_clusters = clusters.copy()

        for c0 in clusters:
            for c1 in clusters[clusters != c0]:
                for c2 in clusters[(clusters != c0) & (clusters != c1)]:
                    c3 = int(clusters[(clusters != c0) & (clusters != c1) & (clusters != c2)])
                    my_new = my.copy()
                    my_new = np.where(my == 10, c0, my_new)
                    my_new = np.where(my == 11, c1, my_new)
                    my_new = np.where(my == 12, c2, my_new)
                    my_new = np.where(my == 13, c3, my_new)
                    accuracy_new = accuracy_score(validation_y, my_new)

                    if accuracy_new > max_accuracy:
                        max_accuracy = accuracy_new
                        true_clusters = np.array([c0, c1, c2, c3])

        return true_clusters, max_accuracy

