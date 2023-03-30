from typing import Tuple
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    Class defining a graph timeseries dataset for self-supervised
    learning.
    
    Attributes
    ----------
    x : ndarray
        The input values of the dataset.
    y : ndarray
        The ground truth of the dataset.
    len : int
        The number of instances of the dataset.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Initialize the dataset

        Parameters
        ----------
        x : ndarray
            The input values of the dataset.
        y : ndarray
            The ground truth of the dataset's input data.
        """
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a dataset 

        Parameters
        ----------
        index : int
            The index from where to extract the dataset instance.

        Returns
        -------
        ndarray
            The input data at the given index.
        ndarray
            The ground truth with respect to the input data at the
            given index.
        """
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        """Get the length of the dataset-

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.len

def get_dataloader(x: np.ndarray, y: np.ndarray, 
                   batch_size: int, shuffle: bool) -> DataLoader:
    """
    Get a dataloader from a series of input data and their ground
    truth values.

    Parameters
    ----------
    x : ndarray
        The input values 
    y : ndarray
        The ground truth of the input data.
    batch_size : int
        The batch size to use for the dataloader.
    shuffle : bool
        Whether to shuffle the data or not while fetching it from
        the dataloader.

    Returns
    -------
    DataLoader
        The dataoader-
    """
    dataset = TimeSeriesDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
