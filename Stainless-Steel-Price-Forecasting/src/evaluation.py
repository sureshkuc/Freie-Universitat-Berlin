"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions for evaluating machine learning model predictions,
    including metrics like Mean Absolute Percentage Error (MAPE), Directional Symmetry,
    and functions for testing the stationarity of time series data.
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: MAPE value, scaled by 100.
    
    Raises:
        ValueError: If `y_true` contains zero values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values, which will cause division by zero.")
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def directional_symmetry(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional symmetry between the true and predicted values.

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: Proportion of matching directional changes between `y_true` and `y_pred`.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    correct_direction = np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
    return correct_direction.mean()


def test_stationarity(timeseries: pd.Series, window: int = 15, cutoff: float = 0.01) -> None:
    """
    Perform the Augmented Dickey-Fuller test to check for stationarity of a time series.

    Args:
        timeseries (pd.Series): The time series data to be tested.
        window (int, optional): The rolling window size for mean and std (default is 15).
        cutoff (float, optional): The p-value threshold for determining stationarity (default is 0.01).

    Returns:
        None: Prints the results of the stationarity test.
    """
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print(f'p-value = {pvalue:.4f}. The series is likely stationary.')
    else:
        print(f'p-value = {pvalue:.4f}. The series is likely non-stationary.')
    print(dfoutput)


def plot_predictions(dataset: np.ndarray, train_predict: np.ndarray, test_predict: np.ndarray) -> None:
    """
    Plot the predictions from the training and test datasets along with the original dataset.

    Args:
        dataset (np.ndarray): The original dataset.
        train_predict (np.ndarray): The model's predictions for the training data.
        test_predict (np.ndarray): The model's predictions for the test data.

    Returns:
        None: Displays the plot.
    """
    look_back = 1
    train_predict_plot = np.empty_like(dataset[:, 0])
    train_predict_plot[:] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back] = train_predict.reshape(train_predict.shape[0])

    test_predict_plot = np.empty_like(dataset[:, 0])
    test_predict_plot[:] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1] = test_predict.reshape(test_predict.shape[0])

    plt.figure(figsize=(20, 10))
    plt.plot(dataset[:, 0], label='Original Data')
    plt.plot(train_predict_plot, label='Train Data Prediction by LSTM')
    plt.plot(test_predict_plot, label='Test Data Prediction by LSTM')
    plt.ylabel('Number')
    plt.legend()
    plt.show()


def main() -> None:
    """
    Main function to demonstrate the usage of the above functions.
    This function will not be called if this module is imported.
    
    Returns:
        None
    """
    # Example usage for testing functions could be placed here, if necessary.
    pass


if __name__ == "__main__":
    main()

