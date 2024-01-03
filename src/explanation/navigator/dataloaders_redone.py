from typing import Tuple
from torch.utils.data.dataloader import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd

from ..events import get_largest_event_set
from ...data.data_processing import Scaler
from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...spatial_temporal_gnn.metrics import MAE


class EventsDataset(Dataset):
    """
    Class defining a events dataset for self-supervised learning training
    of the navigator model.

    Attributes
    ----------
    x : ndarray
        The input values of the dataset.
    y : ndarray
        The ground truth of the dataset.
    target_events : ndarray | None
        The target events of the dataset. None if they are
        built while the dataset is being generated.

    len : int
        The number of instances of the dataset.
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_time: np.ndarray,
        spatial_temporal_gnn: SpatialTemporalGNN,
        scaler: Scaler,
        device: str,
        ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        x : ndarray
            The input values of the dataset.
        y : ndarray
            The ground truth of the dataset's input data.
        fix_y : bool, optional
            Whether to fix the ground truth data by randomly sampling
            just at the beginning and not while data is being generated,
            by default False.
        """
        # Filter out the instances that do not have speed values at all.
        self.x = x[np.any(x[..., 0] != 0, axis=(1, 2))]
        self.y = y[np.any(x[..., 0] != 0, axis=(1, 2))]
        self.y_time = y_time[np.any(x[..., 0] != 0, axis=(1, 2))]

        self.spatial_temporal_gnn = spatial_temporal_gnn
        self.ae_criterion = MAE(apply_average=False)
        self.scaler = scaler
        self.device = device
        
        self.len = self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a dataset 

        Parameters
        ----------
        index : int
            The index from where to extract the dataset instance.

        Returns
        -------
        ndarray
            The input data at the given index with masked random
            features.
        ndarray
            The ground truth with respect to the input data at the
            given index with masked random features.
        """
        # Copy the input and ground truth data at the given index.
        x = self.x[index].copy()
        y = self.y[index].copy()
        y_time = self.y_time[index].copy()
    
        #target_event = self.target_events[index] if self.target_events \
        #    else None
        
        # Get the largest event set of the input instance.
        input_events = get_largest_event_set(x)
        input_events = np.array([i[1:] for i in input_events if i[0] == 0])
        
        # Create a list of instances with removed events.
        xs_with_removed_events = []
        
        target_event = self._get_random_target_graph_encoded_event(y, y_time)

        masked_y = np.zeros_like(y)
        target_timestep, target_node = target_event[0 : 2].astype(int)
        masked_y[target_timestep, target_node, 0] =  y[target_timestep, target_node, 0]
        
        for e in input_events:
            timestep, node = e
            # Remove the speed event at timestep i and node j.
            x_with_removed_event = torch.tensor(x).float().to(device=self.device)
            x_with_removed_event[timestep, node, 0] = 0.

            xs_with_removed_events.append(x_with_removed_event)

        with torch.no_grad():
            simulated_instances_loader = DataLoader(
                xs_with_removed_events,
                batch_size=64,
                shuffle=False)

            simulated_instances_scores = []
            for simulated_batch in simulated_instances_loader:
                simulated_batch = self.scaler.scale(simulated_batch)
                simulated_batch = simulated_batch.float().to(device=self.device)
                y_pred = self.spatial_temporal_gnn(simulated_batch)
                y_pred = self.scaler.un_scale(y_pred)

                # Repeat the target graph y by adding a batch dimension for the y_pred batch size.
                y_repeated = torch.tensor(masked_y, dtype=torch.float32, device=self.device)
                y_repeated = y_repeated.unsqueeze(0).repeat(y_pred.shape[0], 1, 1, 1)

                simulated_instances_scores.append(self.ae_criterion(y_pred, y_repeated).cpu().numpy())
            simulated_instances_scores = np.concatenate(simulated_instances_scores)
            
            max_ae = simulated_instances_scores.max()
            # min_ae = simulated_instances_scores.min()
            
            simulated_instances_scores = (max_ae - simulated_instances_scores) / (max_ae)
            
            scores = np.zeros_like(x[..., 0])
            
            for e, s in zip(input_events, simulated_instances_scores):
                timestep, node = e
                scores[timestep, node] = s

        return x, target_event, scores

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.len
    
    def _get_random_target_graph_encoded_event(
        self,
        y: np.ndarray,
        y_time: np.ndarray
        ) -> np.ndarray:
        """Get a random target graph encoded event.

        Parameters
        ----------
        y : ndarray
            The target graph encoded events.

        Returns
        -------
        ndarray
            The random target graph encoded event.
        """
        # Get the largest event set of the output instance.
        target_events = np.array(get_largest_event_set(y))

        [target_event_idx] = np.random.choice(
            np.arange(len(target_events)), size=1, replace=False)

        target_event = target_events[target_event_idx]
        node_speed = y[target_event[1], target_event[2], 0]
        times = y_time[target_event[1], target_event[2]]

        return np.array([target_event[1], target_event[2], node_speed, *times])

def get_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    y_time: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    scaler: Scaler,
    device: str,
    batch_size: int,
    shuffle: bool
    ) -> DataLoader:
    y_time = _get_encoded_times(y_time)
    dataset = EventsDataset(x, y, y_time, spatial_temporal_gnn, scaler, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def _collate_fn(batch):
    print(batch)
    return batch

def _get_encoded_times(node_times: np.ndarray) -> np.ndarray:
    """
    Add a new feature to the node values numpy array containing
    the day of the week encoded as a one-hot vector.

    Parameters
    ----------
    node_values_np : ndarray
        The numpy array containing the features of the nodes at
        different timestamps.
    node_values_df : DataFrame
        The dataframe containing the speed values of the nodes at
        different timestamps.

    Returns
    -------
    ndarray
        The numpy array containing the features of the nodes at
        different timestamps and the day of the week encoded as a
        one-hot vector.
    """
    # Encode the time of the day.
    b, t, n = node_times.shape
    datetimes = pd.DatetimeIndex(node_times.flatten())
    time_of_the_day = np.array([d.hour * 60 + d.minute for d in datetimes])

    # Get the minimum and maximum values of the selected criteria.
    min_value, max_value = (0, 23 * 60 + 59)
    # Scale the time information between 0 and 1.
    time_of_the_day = (time_of_the_day - min_value) / (max_value - min_value)
    
    # Reshape the time of the day array.
    time_of_the_day = time_of_the_day.reshape(b, t, n, 1)
    
    # Encode the time of the day.
    day_of_the_week = np.array(datetimes.day_of_week)

     # One hot encode the time information.
    day_of_the_week = np.eye(7)[day_of_the_week]
    
    # Reshape the time of the day array.
    day_of_the_week = day_of_the_week.reshape(b, t, n, 7)

    # Concatenate the time of the day and the day of the week.
    encoded_times = np.concatenate((time_of_the_day, day_of_the_week), axis=-1)
    return encoded_times
