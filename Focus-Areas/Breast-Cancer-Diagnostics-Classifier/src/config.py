import os
import logging

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# config.py
class Config:
    DATA_PATH = "C:/Users/MACWAN/Documents/data/wdbc_data.csv"
    COLUMN_NAMES = ['id','diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
                    'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                    'fractal_dimension_se', 'radius_worst', 'texture_worst',
                    'perimeter_worst', 'area_worst', 'smoothness_worst',
                    'compactness_worst', 'concavity_worst', 'concave points_worst',
                    'symmetry_worst', 'fractal_dimension_worst']
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    DROP_COLUMNS = ['concavity_se', 'fractal_dimension_se', 'texture_se']



