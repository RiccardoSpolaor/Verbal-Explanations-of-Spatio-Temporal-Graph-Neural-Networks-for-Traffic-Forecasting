from typing import Tuple
from sklearn.model_selection import train_test_split
import numpy as np

def get_node_values_and_adjacency_matrix(
    node_values_file_path: str, adjacency_matrix_file_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(node_values_file_path), np.load(adjacency_matrix_file_path)

def train_test_val_split(
    node_values: np.ndarray, test_size: float = .2,
    val_size: float = .1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test = train_test_split(node_values, test_size=test_size,
                                       shuffle=False)
    x_train, x_val = train_test_split(x_train, test_size=val_size,
                                      shuffle=False)
    return x_train, x_val, x_test
