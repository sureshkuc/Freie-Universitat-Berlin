"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides data cleaning and preprocessing steps for an actor earnings dataset.
    It corrects formatting issues, synchronizes name fields, handles missing values, and ensures 
    proper data types for numerical analysis.
Version: 1.0
"""

import os
import logging
from typing import Optional
import pandas as pd
import numpy as np

# Setup logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/data_cleaning.log",
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

def load_dataset(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame if successful, None otherwise.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded dataset from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path}: {e}", exc_info=True)
        return None

def clean_actor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and processes actor dataset.

    Args:
        df (pd.DataFrame): Raw DataFrame to process.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        # Fixing separator in row 24
        df.at[24, ['Firstname', 'Lastname']] = df.loc[24, 'Actor'].split(':')
        logging.debug("Fixed separator issue in row 24")

        # Synchronize Actor column with Firstname and Lastname
        name_mismatches = df[df['Actor'] != df['Firstname'] + ' ' + df['Lastname']].index
        df.at[name_mismatches, 'Firstname'] = df.loc[name_mismatches, 'Actor'].str.split(' ').str[0]
        df.at[name_mismatches, 'Lastname'] = df.loc[name_mismatches, 'Actor'].str.split(' ').str[1]
        logging.debug(f"Synchronized names at mismatched rows: {list(name_mismatches)}")

        # Clean enclosing characters in row 18
        df.loc[18] = df.loc[18].astype(str).str.replace("'", '', regex=False)
        logging.debug("Removed enclosing characters from row 18")

        # Normalize 'Price' column (binary 'yes'/'no')
        df['Price'] = df['Price'].apply(
            lambda x: 'no' if pd.isnull(x) or not isinstance(x, str) or not x.endswith('s') else 'yes'
        )
        logging.debug("Normalized 'Price' column")

        # Reorder and drop redundant columns
        df = df[[
            'Firstname', 'Lastname', 'Total Gross', 'Number of Movies',
            'Average per Movie', '#1 Movie', 'Gross', 'Price'
        ]]
        logging.info("Reordered columns and removed redundant 'Actor' column")

        # Check for duplicates
        if df.duplicated().any():
            logging.warning("Duplicates found in dataset")
        else:
            logging.info("No duplicates found")

        # Convert numeric columns to float
        numeric_cols = ['Total Gross', 'Number of Movies', 'Average per Movie', 'Gross']
        df[numeric_cols] = df[numeric_cols].astype(float)

        # Ensure all Gross values are positive
        df['Gross'] = df['Gross'].abs()
        logging.info("Converted numeric columns to float and corrected negative gross values")

        return df

    except Exception as e:
        logging.error("Error during data cleaning: %s", e, exc_info=True)
        return df

def main() -> None:
    """
    Main function to execute the data cleaning pipeline.
    """
    file_path = '/home/sofya/Documents/UNI/ML/ml2020/project3/actor_err.csv'
    df = load_dataset(file_path)

    if df is not None:
        cleaned_df = clean_actor_data(df)
        print(cleaned_df.sum(numeric_only=True))
        print(cleaned_df['Price'].value_counts())
    else:
        logging.critical("Dataset could not be processed because it failed to load.")

if __name__ == '__main__':
    main()

