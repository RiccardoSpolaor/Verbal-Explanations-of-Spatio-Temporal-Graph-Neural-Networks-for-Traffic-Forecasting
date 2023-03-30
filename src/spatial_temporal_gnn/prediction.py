from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from .model import SpatialTemporalGNN
from ..data.data_processing import Scaler

def get_ground_truth_and_predictions(
    model: SpatialTemporalGNN, dataloader: DataLoader, device: str,
    scaler: Scaler, n_timestamps_to_predict: Optional[int] = None,
    use_standardized_scale: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the ground truth and predictions of a Spatial-Temporal GNN
    model for a given dataloader.

    Parameters
    ----------
    model : SpatialTemporalGNN
        The Spatial-Temporal GNN model to use for making the predictions.
    dataloader : DataLoader
        The dataloader containing the input data.
    device : str
        The device to use for computing the predictions (e.g. "cpu"
        or "cuda").
    scaler : Scaler
        The stcaler object to use for scaling the input and output data.
    n_timestamps_to_predict : int, optional
        The number of timestamps to predict. If not None, only the first 
        `n_timestamps_to_predict` timestamps will be predicted.
        By default None.
    use_standardized_scale : bool, optional
        If True, the output data will be scaled using the standardized
        scaling instead of the original scaling, by default False.

    Returns
    -------
    ndarray
        The ground truth values as a numpy array. 
    ndarray
        The respective predicted values as a numpy array.
    """
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

def plot_results_comparison(
    y_true: np.ndarray, y_pred: np.ndarray, feature_names: List[str],
    n_random_instances: int = 100, n_random_nodes: int = 3) -> None:
    assert len(feature_names) == y_pred.shape[-1] == y_true.shape[-1], \
    'The feature names length must be the same as the one of the' + \
    'features of the ground truth instances and the predictions.'

    plt.figure(figsize=(15, 10))

    for i, l in enumerate(feature_names):
        plt.subplot(1, 1, i + 1)

        plt.title(f'Comparison of the results for feature {l}')

        random_instances = sorted(
            np.random.choice(range(y_true.shape[0]), n_random_instances,
                             replace=False))
        random_nodes = sorted(
            np.random.choice(range(y_true.shape[2]), n_random_nodes,
                             replace=False))

        y_true_raveled = \
            y_true[random_instances, :][:, :, random_nodes, i].ravel()
        y_pred_reveled = \
            y_pred[random_instances, :][:, :, random_nodes, i].ravel()

        filtered = y_true_raveled > 0
        y_true_raveled = y_true_raveled[filtered]
        y_pred_reveled = y_pred_reveled[filtered]

        plt.plot(y_true_raveled, label='Ground truth')
        plt.plot(y_pred_reveled, label='Predictions')

        plt.xlabel('Instance')
        plt.ylabel(l)

        plt.legend()

    plt.suptitle(
        'Result comparison between the ground truth and the predictions',
        size=16)

    plt.tight_layout()
    plt.show()

def plot_results_fitness(
    y_true: np.ndarray, y_pred: np.ndarray, feature_names: List[str],
    n_random_instances: int = 100, n_random_nodes: int = 50) -> None:
    plt.figure(figsize=(15, 10))
    assert len(feature_names) == y_pred.shape[-1] == y_true.shape[-1], \
    'The feature names length must be the same as the one of the' + \
    'features of the ground truth instances and the predictions'

    for i, l in enumerate(feature_names):
        plt.subplot(1, 1, i + 1)

        plt.title(f'Comparison of the fitness results for feature {l}')

        random_instances = sorted(
            np.random.choice(range(y_true.shape[0]), n_random_instances,
                             replace=False))
        random_nodes = sorted(
            np.random.choice(range(y_true.shape[2]), n_random_nodes,
                             replace=False))

        y_true_raveled = \
            y_true[random_instances, :][:, :, random_nodes, i].ravel()
        y_pred_reveled = \
            y_pred[random_instances, :][:, :, random_nodes, i].ravel()

        filtered = y_true_raveled > 0
        y_true_raveled = y_true_raveled[filtered]
        y_pred_reveled = y_pred_reveled[filtered]

        plt.scatter(y_true_raveled, y_pred_reveled)

        plt.axline((1., 1.), slope=1, color='r', linestyle='--')

        plt.xlabel('Ground truth')
        plt.ylabel('Prediction')

    plt.suptitle(
        'Result fitness comparison between the ground truth and the ' +\
        'predictions', size=16)

    plt.tight_layout()
    plt.show()
