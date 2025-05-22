"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to estimate and compare the variance
    of the sample mean using simple random sampling (SRS) without replacement
    and stratified sampling, applied on FIFA player data.
Version: 1.0
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class FIFADataSampler:
    """Class to handle loading and sampling operations on the FIFA dataset."""

    def __init__(self, filepath: str) -> None:
        """Initialize the data sampler with the given file path.

        Args:
            filepath (str): Path to the CSV file containing FIFA data.
        """
        self.df = pd.read_csv(filepath, index_col=0)
        self.N = self.df.shape[0]

    def analyze_missing_data(self) -> pd.Series:
        """Analyze missing data per column.

        Returns:
            pd.Series: Number of missing values per column.
        """
        attributes = len(self.df) - self.df.count(axis=0)
        return attributes

    def sample_variance_mean_srs(self, column: str, sample_sizes: List[int], iterations: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute sampling variance and mean using Simple Random Sampling.

        Args:
            column (str): Name of the column to sample.
            sample_sizes (List[int]): List of sample sizes.
            iterations (int, optional): Number of iterations. Defaults to 100.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of variances and means.
        """
        X = self.df[column]
        sample_var = pd.DataFrame(index=sample_sizes)
        sample_mean = pd.DataFrame(index=sample_sizes)

        for n in sample_sizes:
            for i in range(iterations):
                sample = X.sample(n, replace=False, random_state=i)
                sample_var.at[n, i] = (sample.var() / n) * (1 - (n - 1) / (self.N - 1))
                sample_mean.at[n, i] = sample.mean()

        return sample_var, sample_mean

    def sample_variance_mean_stratified(self, groupby_column: str, target_column: str, sampling_ratio: float, iterations: int = 100) -> Tuple[List[float], List[float]]:
        """Compute sampling variance and mean using Stratified Sampling.

        Args:
            groupby_column (str): Column to stratify on.
            target_column (str): Column for which to compute statistics.
            sampling_ratio (float): Fraction of data to sample.
            iterations (int, optional): Number of iterations. Defaults to 100.

        Returns:
            Tuple[List[float], List[float]]: Lists of variances and means.
        """
        sample_var = []
        sample_mean = []

        for i in range(iterations):
            sample = self.df.groupby(groupby_column).apply(
                lambda x: x.sample(frac=sampling_ratio, random_state=i)
            ).reset_index(drop=True)
            sample_var.append((sample[target_column].var() / sample.shape[0]) * (1 - (sample.shape[0] - 1) / (self.N - 1)))
            sample_mean.append(sample[target_column].mean())

        return sample_var, sample_mean

    def plot_variances(self, variances: pd.DataFrame, means: pd.DataFrame, title_prefix: str) -> None:
        """Plot sampling variances and variance of sample means.

        Args:
            variances (pd.DataFrame): DataFrame of sample variances.
            means (pd.DataFrame): DataFrame of sample means.
            title_prefix (str): Prefix for subplot titles.
        """
        plt.figure(figsize=(16, 9))
        for i, sample_size in enumerate(variances.index):
            plt.subplot(1, len(variances.index), i + 1)
            plt.scatter(range(1, 101), variances.loc[sample_size].values)
            plt.axhline(variances.loc[sample_size].mean())
            plt.axhline(means.var(axis=1).loc[sample_size], color='red')
            plt.xlabel("Sample #")
            if i == 0:
                plt.ylabel("Variance")
            plt.title(f"{title_prefix} {sample_size} samples")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    cwd = os.getcwd()
    filepath = os.path.join(cwd, "fifa.csv")
    sampler = FIFADataSampler(filepath)

    sample_sizes = [15000, 10000, 5000, 1000, 100]

    # SRS for 'Finishing'
    var_srs, mean_srs = sampler.sample_variance_mean_srs("Finishing", sample_sizes)
    sampler.plot_variances(var_srs, mean_srs, "SRS")

    # Stratified sampling by 'Position'
    var_strat, mean_strat = sampler.sample_variance_mean_stratified("Position", "Finishing", 1 / 5)

    # Additional plots or exports can be done similarly
    print("SRS Mean Variance by Sample Size:")
    print(mean_srs.var(axis=1))

