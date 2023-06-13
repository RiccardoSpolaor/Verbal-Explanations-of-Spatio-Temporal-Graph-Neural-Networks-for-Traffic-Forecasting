from typing import Tuple
from sklearn.model_selection import train_test_split
import numpy as np

def train_test_val_split(
    node_values: np.ndarray,
    node_times: np.ndarray,
    test_size: float = .2,
    val_size: float = .1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the train, validation and test split of a dataset.

    Parameters
    ----------
    node_values : ndarray
        The node values dataset to split in train test and validation
    node_times : ndarray
        The node times dataset to split in train test and validation
    test_size : float, optional
        The test ratio, by default 0.2.
    val_size : float, optional
        The validation ratio, by default 0.1.

    Returns
    -------
    ndarray
        The train dataset.
    ndarray
        The validation dataset.
    ndarray
        The test dataset.
    ndarray
        The train times dataset.
    ndarray
        The validation times dataset.
    ndarray
        The test times dataset.
    """
    x_train, x_test, train_times, test_times = train_test_split(
        node_values, node_times, test_size=test_size, shuffle=False)
    x_train, x_val, train_times, val_times = train_test_split(
        x_train, train_times, test_size=val_size, shuffle=False)
    return x_train, x_val, x_test, train_times, val_times, test_times
