from typing import Tuple
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return self.len

def get_dataloader(x: np.ndarray, y: np.ndarray, 
                   batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TimeSeriesDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
