"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to perform price prediction using different machine learning models such as LSTM, XGBoost, Random Forest, and Linear Regression.
    It loads, preprocesses, and trains the models, then evaluates them using metrics like Mean Absolute Percentage Error (MAPE) and Directional Symmetry. 
    The results are plotted for comparison.
Version: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from model import PricePredictionModel
from evaluation import mean_absolute_percentage_error, directional_symmetry


def load_and_preprocess_data(url: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset from a CSV file.

    Args:
        url (str): URL of the CSV file to load.

    Returns:
        pd.DataFrame: Processed data with an additional column for the number of days from today.
    """
    data = pd.read_csv(url)
    data['dd'], data['mm'], data['yyyy'] = data['Date'].str.split("/", expand=True)[0], \
                                              data['Date'].str.split("/", expand=True)[1], \
                                              data['Date'].str.split("/", expand=True)[2]
    today = date.today()
    number_of_days = []
    dates = []
    for yy, mm, dd in zip(data['yyyy'], data['mm'], data['dd']):
        d1 = date(int(yy), int(mm), int(dd))
        dates.append(d1)
        delta = today - d1
        number_of_days.append(delta.days)
    data['Date'] = dates
    data['number_of_days_from_today'] = number_of_days
    return data


def prepare_train_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare training data by renaming and dropping unnecessary columns.

    Args:
        data (pd.DataFrame): Preprocessed data.

    Returns:
        pd.DataFrame: Data prepared for training with 'ds' and 'y' columns.
    """
    X = data.drop(columns=['dd', 'mm', 'yyyy'])
    return X.rename(columns={"Date": "ds", "StainlessSteelPrice": "y"})


def plot_results(y_pred: pd.DataFrame, data_train: pd.DataFrame) -> None:
    """
    Plot the predicted vs actual values.

    Args:
        y_pred (pd.DataFrame): The predicted values.
        data_train (pd.DataFrame): The actual values.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(y_pred['ds'], y_pred['yhat'], label='Predicted Price')
    plt.plot(data_train['ds'], data_train['y'], label='Actual Price')
    plt.legend()
    plt.title('Price Prediction vs Actual')
    plt.show()


def main():
    """
    Main function to load, preprocess, train the model, and evaluate the results.
    """
    # Load and preprocess the data
    url = 'https://raw.githubusercontent.com/sureshkuc/chemovator./main/Stainless-Steel-Prices-Forecasty-Assignment.csv'
    data = load_and_preprocess_data(url)
    
    # Prepare train data
    data_train = prepare_train_data(data)

    # Define the parameter grid for tuning
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    # Initialize the model
    model = PricePredictionModel(param_grid)
    model.train(data_train)

    # Forecasting
    y_pred = model.forecast(data_train)
    plot_results(y_pred, data_train)

    # Evaluate the model performance
    print("MAPE Mean Absolute Percentage Error:", mean_absolute_percentage_error(data_train['y'], y_pred['yhat']))
    print("Directional Symmetry Statistic:", directional_symmetry(data_train['y'], y_pred['yhat']))


if __name__ == "__main__":
    main()


