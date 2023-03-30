from typing import get_args, List, Literal, Tuple

import torch
import numpy as np
import pandas as pd

TimeAggregation = Literal['time_of_day', 'day_of_week', 'hour', 'minute',
                          'day_of_year']
TIME_AGGREGATION_TUPLE = get_args(TimeAggregation)

class Scaler():
    """
    Class defining a scaler to apply standard scaling individually
    to each feature of a dataset or some instancrs of a dataset. 
    The scaling is based on the mean and standard deviation computed
    on the dataset used for initialization.
    
    Attributes
    ----------
    mean : FloatTensor
        The means used for standard scaling. They are computed on the
        datset provided for initialization. Each mean is defined for a
        specific feature of the dataset.
    std : FloatTensor
        The standard deviations used for standard scaling. They are 
        computed on the datset provided for initialization. Each 
        standard deviation is defined for a specific feature of
        the dataset.
    """
    def __init__(self, x: torch.FloatTensor) -> None:
        """Initialize a `Scaler` instance.

        Parameters
        ----------
        x : FloatTensor
            The dataset on which the mean and standard deviation
            are computed as estimations to apply standard scaling
            to future instances of the same kind.
        """
        with torch.no_grad():
            x = x.clone()
            self.mean = torch.mean(x, dim=(-3, -2))
            self.std = torch.std(x, dim=(-3, -2))

    def scale(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Scale the provided dataset or series instances through
        standard scaling.

        Parameters
        ----------
        x : FloatTensor
            The dataset or series of instances to scale.

        Returns
        -------
        FloatTensor
            The resulting scaled dataset or series of instances.
        """
        x = x.clone()
        n_features = x.shape[-1]
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x = (x - mean[:n_features]) / std[:n_features]
        return x

    def un_scale(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Bring a dataset or a series of instances that have
        been previously scaled through standard scaling to their
        original non-scaled representation.

        Parameters
        ----------
        x : FloatTensor
            The dataset or series of instances to un-scale.

        Returns
        -------
        FloatTensor
            The resulting un-scaled dataset or series of instances.
        """
        x = x.clone()
        n_features = x.shape[-1]
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x = x * std[:n_features] + mean[:n_features]
        return x

def get_node_values_numpy_matrix(
    node_values_df: pd.DataFrame, time_aggregations: List[TimeAggregation]
    ) -> np.ndarray:
    """
    Get a numpy array containing the speed values of the nodes
    at different timestamps from a specific dataframe.
    The returning array will have the following shape (T, N, F), where:
    * T is the number of timesteps.
    * N is the number of nodes.
    * F is the number of features.
    F varies between 1 or more according on how many `time_encodings`
    elements are provided.

    Parameters
    ----------
    node_values_df : DataFrame
        The dataframe containing the speed values of the nodes at
        different timestamps.
    time_aggregations : set of 'day_of_week' | 'hour' | 'minute' |
    'day_of_year'
        The set of methods by which the time features will be
        aggregated and encoded.

    Returns
    -------
    ndarray
        The array containing the speed values of the nodes and the eventual
        time encoded information.
    """
    # Get the numpy matrix of node values from the pandas dataframe.
    node_values_np = node_values_df.to_numpy()
    # Expand the dimension of the dataframe to (T, N, F)
    node_values_np = np.expand_dims(node_values_df.to_numpy(), axis=-1)

    _, _, n_features = node_values_np.shape

    for i, t in enumerate(time_aggregations):
        encoded_time = _get_encoded_time_information(node_values_df,
                                                     aggregate_by=t)
        node_values_np = np.insert(node_values_np, n_features + i,
                                   values=encoded_time, axis=-1)
    return node_values_np

def _get_encoded_time_information(
    node_values_df: pd.DataFrame,
    aggregate_by: TimeAggregation
    ) -> np.ndarray:
    """
    Get the encoded time information at each timestamp for each node
    aggregated according to the `aggregate_by` feature.

    Parameters
    ----------
    node_values_df : DataFrame
        The dataframe containing the speed values of the nodes at different
        timestamps.
    aggregate_by : 'day_of_week' | 'hour' | 'minute' | 'day_of_year'
        The method by which the timestamps will be aggregated and then
        encoded.
        * 'day_of_week': encodes the timestamps by day of the week [0-6].
        * 'hour': encodes the timestamps by hour of the day [0-23].
        * 'minute': encodes the timestamps by minute of the hour [0-59].
        * 'day_of_year': encodes the timestamps by day of the year [0-365]. 

    Returns
    -------
    ndarray
        The time feature for each node at each timestamp aggregated
        according to `aggregate_by` and properly encoded in a value
        between 0 and 1. 
    """
    # Assert that `aggregate_by` is provided correctly.
    assert aggregate_by in TIME_AGGREGATION_TUPLE, \
        'The `aggregate_by` attribute should be one among ' +\
        f'{"; ".join(TIME_AGGREGATION_TUPLE)}'

    # Set the dictionary of grouping criteria.
    grouping_criteria = {
        'time_of_day': [d.hour * 60 + d.minute 
                        for d in pd.to_datetime(node_values_df.index.values)],
        'day_of_week': node_values_df.index.day_of_week,
        'hour': node_values_df.index.hour,
        'minute': node_values_df.index.minute,
        'day_of_year': node_values_df.index.day_of_year
    }

    #Set the dictionary of minimum and maximum values by criteria.
    min_max_by_criteria = {
        'time_of_day': (0, 23 * 60 + 59),
        'day_of_week': (0, 6),
        'hour': (0, 23),
        'minute': (0, 59),
        'day_of_year': (0, 366)
    }

    # Get the time information by criteria.
    time_information = np.array(grouping_criteria[aggregate_by])

    # Get the minimum and maximum values of the selected criteria.
    min_value, max_value = min_max_by_criteria[aggregate_by]
    # Scale the time information between 0 and 1.
    time_information = (time_information - min_value) / (max_value - min_value)

    # Get the number of nodes in the dataframe.
    _, n_nodes = node_values_df.shape

    # Add a dimension to the time information.
    time_information = np.expand_dims(time_information, axis=-1)
    # Repeat for each node the time information at a specific timestamp.
    time_information = np.repeat(time_information, n_nodes, axis = 1)

    return time_information

def get_dataset_by_sliding_window(
    dataset: np.ndarray, x_stepsize: int, y_stepsize: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a sliding window of size (`x_stepsize` + `y_stepsize`)
    to the input dataset in order to obtain a division of it (`x`, `y`)
    where:
    * The first `x_stepsize` elements captured by the window are
    assigned to the `x` dataset.
    * The last `y_stepsize` elements captured by the window are
    assigned to the `y` dataset.
    The obtained division dataset can be used for self-supervised
    learning activities, as the first division (`x`) can be used
    as input elements, while the second division (`y`) can be used
    as ground truth results.

    Parameters
    ----------
    dataset : ndarray
        The input numpy array that will be divided by the sliding
        window.
    x_stepsize : int
        The number of first elements of the window that will be
        assigned to the first division of the dataset (`x`).
    y_stepsize : int
        The number of last elements of the window that will be
        assigned to the second division of the dataset (`y`).

    Returns
    -------
    ndarray
        The first division of the dataset.
    ndarray
        The second division of the dataset.
    """
    # Initialize the dataset divisions as empty lists.
    x, y = [], []
    # Initialize the iteration count at zero.
    i = 0

    while True:
        # Get the current index position of the sliding window on the
        # dataset.
        index = y_stepsize * i + x_stepsize * i

        # Stop the iteration if the sliding window overflows the dataset.
        if index + x_stepsize + y_stepsize >= len(dataset):
            break

        # Apply the sliding window on the data and assign the result
        # on the respective divisions.
        x.append(dataset[index : index + x_stepsize])
        y.append(dataset[index + x_stepsize : index + x_stepsize + y_stepsize])

        # Increase the iteration count.
        i += 1

    # Stack the results.
    return np.stack(x), np.stack(y)
