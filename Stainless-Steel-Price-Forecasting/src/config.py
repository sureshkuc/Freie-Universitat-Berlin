"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes for preprocessing data, setting random seeds,
    and initializing scalers for use in machine learning models, including XGBoost, ARIMA, and Prophet.
Version: 1.0
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston, load_diabetes
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost
from xgboost import plot_importance
from fbprophet import Prophet
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import performance_metrics, cross_validation
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
import torch
import torch.nn as nn
import itertools
from collections import OrderedDict

%matplotlib inline
%load_ext autoreload
%autoreload 2
register_matplotlib_converters()

# Set random seed for reproducibility
def set_random_seed(seed: int = 7) -> None:
    """
    Sets the random seed for NumPy and PyTorch to ensure reproducibility of results.
    
    Args:
        seed (int): The seed value for random number generation. Default is 7.

    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

# Initialize scalers
def get_scalers() -> tuple:
    """
    Initializes and returns two scalers: MinMaxScaler and StandardScaler.

    Returns:
        tuple: A tuple containing MinMaxScaler and StandardScaler objects.
    """
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    standard_scaler = StandardScaler()
    return min_max_scaler, standard_scaler

