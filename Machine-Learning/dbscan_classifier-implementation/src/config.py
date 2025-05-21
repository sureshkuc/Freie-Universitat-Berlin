"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module sets up logging, creates necessary output directories,
    and configures environment limits for a data processing or clustering pipeline.
    It includes error handling and logs all activity to a file.

Version: 1.0
"""

import os
import sys
import logging
from typing import Optional


def create_output_directory(directory: str = "outputs") -> Optional[str]:
    """
    Create the output directory if it does not exist.

    Args:
        directory (str): The name of the output directory. Defaults to "outputs".

    Returns:
        Optional[str]: The path to the output directory if created successfully, None otherwise.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logging.debug("Output directory created or already exists: %s", directory)
        return directory
    except OSError as e:
        logging.error("Failed to create output directory: %s", e)
        return None


def configure_logging(log_dir: str = "outputs", log_filename: str = "dbscan_pipeline.log") -> None:
    """
    Configure the logging settings for the application.

    Args:
        log_dir (str): Directory where the log file should be saved.
        log_filename (str): Name of the log file.
    """
    try:
        log_path = os.path.join(log_dir, log_filename)

        logging.basicConfig(
            filename=log_path,
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        logging.info("Logging configured successfully. Log file: %s", log_path)

    except Exception as e:
        print(f"Error setting up logging: {e}")


def set_recursion_limit(limit: int = 10**6) -> None:
    """
    Set the maximum recursion limit for the Python interpreter.

    Args:
        limit (int): The recursion limit to be set. Default is 1,000,000.
    """
    try:
        sys.setrecursionlimit(limit)
        logging.info("Recursion limit set to %d", limit)
    except ValueError as e:
        logging.error("Failed to set recursion limit: %s", e)


def initialize_environment() -> None:
    """
    Initialize the environment by setting up the output directory,
    configuring logging, and setting the recursion limit.
    """
    try:
        output_path = create_output_directory()
        if output_path:
            configure_logging(output_path)
        else:
            raise RuntimeError("Output directory could not be created.")

        set_recursion_limit()

        logging.info("Environment initialization complete.")

    except Exception as e:
        logging.critical("Environment initialization failed: %s", e)
        raise


if __name__ == "__main__":
    initialize_environment()

