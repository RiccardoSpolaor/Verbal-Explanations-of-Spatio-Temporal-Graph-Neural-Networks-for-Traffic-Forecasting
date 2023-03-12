from typing import Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
import numpy as np

from .model import SpatialTemporalGNN
from ..data.data_processer import Scaler

def get_ground_truth_and_predictions(
    model: SpatialTemporalGNN, dataloader: DataLoader, device: str,
    scaler: Scaler, n_timestamps_to_predict: Optional[int] = None,
    use_standardized_scale: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    torch.cuda.empty_cache()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            # Get the data.
            x, y = data
            x = scaler.scale(x)
            x = x.type(torch.float32).to(device=device)
            y = y.type(torch.float32).to(device=device)

            # Compute output.
            y_pred = model(x)

            if n_timestamps_to_predict is not None:
                y_pred = y_pred[:, : n_timestamps_to_predict]
                y = y[:, : n_timestamps_to_predict]

            if use_standardized_scale:
                y = scaler.scale(y)
            else:
                y_pred = scaler.un_scale(y_pred)
            y_true_list.extend(y.cpu().numpy())
            y_pred_list.extend(y_pred.cpu().numpy())

    torch.cuda.empty_cache()

    return np.array(y_true_list), np.array(y_pred_list)

def predict(model: SpatialTemporalGNN, x: Union[np.ndarray, torch.FloatTensor], 
            scaler: object, device: str,
            n_timestamps_to_predict: Optional[int]) -> np.ndarray:
    torch.cuda.empty_cache()

    with torch.no_grad():
        x_shape = x.shape
        if type(x) is np.ndarray:
            x = torch.tensor(x, type=np.float32)

        if len(x_shape == 3):
            x = torch.unsqueeze(x, dim=0)

        x = scaler.scale(x)
        x = x.to(device)

        # Compute output.
        y_pred = model(x)

        if n_timestamps_to_predict is not None:
            y_pred = y_pred[:, : n_timestamps_to_predict]

        y_pred = scaler.un_scale(y_pred)

        if len(x_shape == 3):
            y_pred = torch.squeeze(y_pred, dim=0)

    torch.cuda.empty_cache()
    return y_pred.cpu().numpy()
