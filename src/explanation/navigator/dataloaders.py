from typing import Tuple
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np
import pandas as pd

from ..events import get_largest_event_set


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
        fix_y: bool = False
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
        
        #if fix_y:
        #    target_events = [
        #        self._get_random_target_graph_encoded_event(y_) for y_ in y]
        #    self.target_events = np.array(target_events)
        #else:
        #    self.target_events = None

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
        input_events = np.array(get_largest_event_set(x))

        # Set the instances list.
        instances = []
        for event in input_events:
            # Get the encoded information depending on the event kind.
            event_kind = event[0]
            if event_kind == 0:
                encoded_information = x[event[1], event[2], :]
                ###
                instance = [event[1], event[2], *encoded_information]
                instances.append(instance)
                #encoded_information[1] = -1
                #encoded_information[-7:] = 0
            '''elif event_kind == 1:
                encoded_information = x[event[1], 0, :]
                encoded_information[0] = 0
                encoded_information[-7:] = 0
            elif event_kind == 2:
                encoded_information = x[0, 0, :]
                encoded_information[0] = 0
                encoded_information[1] = -1
            # Encode the timestep and node information.
            timestep = event[1] if event_kind in [0, 1] else -1
            node = event[2] if event_kind == 0 else -1
            # Build the encoded instance and append it to the instances list.
            instance = [event[0], timestep, node, *encoded_information]
            instances.append(instance)''';
        instances = np.array(instances)

        #if not target_event:
        target_event = self._get_random_target_graph_encoded_event(y, y_time)

        masked_y = np.zeros_like(y)
        target_timestep, target_node = target_event[0 : 2].astype(int)
        masked_y[target_timestep, target_node, 0] =\
            y[target_timestep, target_node, 0]

        return x, instances, target_event, masked_y

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
    batch_size: int, shuffle: bool
    ) -> DataLoader:
    y_time = _get_encoded_times(y_time)
    dataset = EventsDataset(x, y, y_time)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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
