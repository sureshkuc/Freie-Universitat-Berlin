"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to perform time series forecasting 
    using different machine learning models, including Prophet, XGBoost, Random Forest, 
    Linear Regression, and AutoFeat. The models are used for training and forecasting 
    with appropriate hyperparameter tuning.
Version: 1.0
"""

import itertools
import numpy as np
from datetime import date
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from autofeat import AutoFeatRegressor

class PricePredictionModel:
    """
    A class used to represent a price prediction model.

    Attributes
    ----------
    param_grid : dict
        A dictionary of hyperparameters for tuning the model.
    best_params : dict
        The best hyperparameters selected based on model performance.
    mape : list
        A list of mean absolute percentage errors for different hyperparameter combinations.

    Methods
    -------
    train(data_train):
        Trains the Prophet model using the provided training data and performs hyperparameter tuning.
    forecast(data_train):
        Forecasts future values using the best model obtained during training.
    """

    def __init__(self, param_grid: dict):
        """
        Initializes the price prediction model with the given parameter grid.

        Parameters
        ----------
        param_grid : dict
            A dictionary containing hyperparameter values for tuning the Prophet model.
        """
        self.param_grid = param_grid
        self.best_params = None
        self.mape = []

    def train(self, data_train: pd.DataFrame):
        """
        Trains the Prophet model using the training data and performs hyperparameter tuning.

        Parameters
        ----------
        data_train : pd.DataFrame
            The training data containing time series data with columns ['ds', 'y'] and optional regressors.
        """
        all_params = [dict(zip(self.param_grid.keys(), v)) for v in itertools.product(*self.param_grid.values())]
        for params in all_params:
            model = Prophet(**params)
            for col in data_train.columns:
                if col not in ['ds', 'y']:
                    model.add_regressor(col)
            model.fit(data_train)
            cutoffs = pd.to_datetime(['2013-01-10', '2014-01-09', '2015-01-09', '2016-01-09', '2017-01-09', 
                                      '2018-01-09', '2019-01-09'])
            df_cv = cross_validation(model, cutoffs=cutoffs, horizon='2 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            self.mape.append(df_p['mape'].values[0])
        
        tuning_results = pd.DataFrame(all_params)
        tuning_results['mape'] = self.mape
        self.best_params = all_params[np.argmin(self.mape)]

    def forecast(self, data_train: pd.DataFrame) -> pd.DataFrame:
        """
        Forecasts future values using the best model obtained during training.

        Parameters
        ----------
        data_train : pd.DataFrame
            The training data containing time series data with columns ['ds', 'y'] and optional regressors.

        Returns
        -------
        pd.DataFrame
            The forecasted values for future periods.
        """
        model = Prophet(changepoint_prior_scale=self.best_params['changepoint_prior_scale'],
                        seasonality_prior_scale=self.best_params['seasonality_prior_scale'])
        for col in data_train.columns:
            if col not in ['ds', 'y']:
                model.add_regressor(col)
        model.fit(data_train)
        future = model.make_future_dataframe(periods=180)
        return model.predict(future)


def build_lstm_model(X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 1, epochs: int = 1000) -> Sequential:
    """
    Builds and trains an LSTM model for time series forecasting.

    Parameters
    ----------
    X_train : np.ndarray
        The training feature data.
    y_train : np.ndarray
        The training target data.
    batch_size : int, optional
        The batch size for training (default is 1).
    epochs : int, optional
        The number of epochs for training (default is 1000).

    Returns
    -------
    Sequential
        The trained LSTM model.
    """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)
    return model


def xgboost_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Trains an XGBoost model with hyperparameter tuning.

    Parameters
    ----------
    X_train : np.ndarray
        The training feature data.
    y_train : np.ndarray
        The training target data.
    X_test : np.ndarray
        The test feature data.
    y_test : np.ndarray
        The test target data.

    Returns
    -------
    xgboost.XGBRegressor
        The best XGBoost model after hyperparameter tuning.
    """
    n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 100, 500, 900, 1100, 1500]
    max_depth = [2, 3, 5, 10, 15]
    booster = ['gbtree', 'gblinear']
    base_score = [0.25, 0.5, 0.75, 1]
    learning_rate = [0.001, 0.01, 0.05, 0.1, 0.15, 0.20, 1]
    min_child_weight = [1, 2, 3, 4]

    hyperparameter_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'min_child_weight': min_child_weight,
        'booster': booster,
        'base_score': base_score
    }

    mape = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    regressor = xgboost.XGBRegressor()
    random_cv = RandomizedSearchCV(estimator=regressor, param_distributions=hyperparameter_grid, cv=5,
                                  n_iter=50, scoring=mape, n_jobs=4, verbose=5, return_train_score=True, random_state=42)
    random_cv.fit(X_train, y_train)
    return random_cv.best_estimator_


def random_forest_model(X_train: np.ndarray, y_train: np.ndarray):
    """
    Trains a Random Forest Regressor model.

    Parameters
    ----------
    X_train : np.ndarray
        The training feature data.
    y_train : np.ndarray
        The training target data.

    Returns
    -------
    RandomForestRegressor
        The trained Random Forest model.
    """
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def linear_regression_model(X_train: np.ndarray, y_train: np.ndarray):
    """
    Trains a Linear Regression model.

    Parameters
    ----------
    X_train : np.ndarray
        The training feature data.
    y_train : np.ndarray
        The training target data.

    Returns
    -------
    LinearRegression
        The trained Linear Regression model.
    """
    model = LinearRegression().fit(X_train, y_train)
    return model


def autofeat_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, feateng_steps: int = 2, units: int = None):
    """
    Trains an AutoFeat model for feature engineering and regression.

    Parameters
    ----------
    X_train : np.ndarray
        The training feature data.
    y_train : np.ndarray
        The training target data.
    X_test : np.ndarray
        The test feature data.
    y_test : np.ndarray
        The test target data.
    feateng_steps : int, optional
        The number of feature engineering steps (default is 2).
    units : int, optional
        The number of units (default is None).

    Returns
    -------
    AutoFeatRegressor
        The trained AutoFeat model.
    np.ndarray
        The transformed training data.
    np.ndarray
        The transformed test data.
    """
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=feateng_steps, units=units)
    X_train_tr = afreg.fit_transform(X_train, y_train)
    X_test_tr = afreg.transform(X_test)
    return afreg, X_train_tr, X_test_tr

