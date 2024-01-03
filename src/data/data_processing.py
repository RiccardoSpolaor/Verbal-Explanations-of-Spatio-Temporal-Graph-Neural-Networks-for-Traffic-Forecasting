from typing import get_args, Dict, List, Literal, Tuple

import torch
import networkx as nx
import numpy as np
import pandas as pd
from geopy.distance import distance as haversine

TimeAggregation = Literal['time_of_day', 'day_of_week']
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
    def __init__(self, x: torch.FloatTensor, has_day_of_week: bool) -> None:
        """Initialize a `Scaler` instance.

        Parameters
        ----------
        x : FloatTensor
            The dataset on which the mean and standard deviation
            are computed as estimations to apply standard scaling
            to future instances of the same kind.
        has_day_of_week : bool
            Whether the dataset contains the day of the week as
            a one-hot encoded feature or not.
        """
        self.has_day_of_week = has_day_of_week

        with torch.no_grad():
            x = x.clone()
            self.mean = torch.mean(x, dim=(-3, -2))
            self.std = torch.std(x, dim=(-3, -2))
            # If the dataset contains the day of the week as a one-hot
            # encoded feature, then make sure that they are not scaled.
            if has_day_of_week:
                self.mean[-7:] = 0.
                self.std[-7:] = 1.
                

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
    time_aggregations : list of 'time_of_day' | 'day_of_week'
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
    times_np = node_values_df.index.values
    # Expand the dimension of the dataframe to (T, N, F)
    node_values_np = np.expand_dims(node_values_df.to_numpy(), axis=-1)

    for t in time_aggregations:
        if t == 'time_of_day':
            node_values_np = _add_encoded_time_of_the_day(
                node_values_np, node_values_df)
        elif t == 'day_of_week':
            node_values_np = _add_encoded_day_of_the_week(
                node_values_np, node_values_df)

    # Add the timesteps as a last feature.
    # Get the number of nodes in the dataframe.
    _, n_nodes = node_values_df.shape

    # Create a new column containing the time information.
    node_times = np.expand_dims(node_values_df.index.values, axis=-1)
    # Repeat for each node the time information at a specific timestamp.
    node_times = np.repeat(node_times, n_nodes, axis=1)

    return node_values_np, node_times

def get_node_values_numpy_matrix_for_data_analysis(
    node_values_df: pd.DataFrame
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
    time_aggregations : list of 'time_of_day' | 'day_of_week'
        The set of methods by which the time features will be
        aggregated and encoded.

    Returns
    -------
    ndarray
        The array containing the speed values of the nodes and the eventual
        time encoded information.
    """
    # Get the numpy matrix of node values from the pandas dataframe.
    node_values_np = node_values_df.to_numpy(dtype=object)
    # Expand the dimension of the dataframe to (T, N, F)
    node_values_np = np.expand_dims(node_values_df.to_numpy(), axis=-1)
    
    _, n_nodes = node_values_df.shape

    time_of_the_day = np.array(
        [f'{d.hour:02d}:{d.minute:02d}'
         for d in pd.to_datetime(node_values_df.index.values)])
    day_of_the_week = np.array([d.day_name() for d in pd.to_datetime(node_values_df.index.values)])
    
    # Add a dimension to the time of the day array.
    time_of_the_day = np.expand_dims(time_of_the_day, axis=-1)
    # Repeat for each node the time information at a specific timestamp.
    time_of_the_day = np.repeat(time_of_the_day, n_nodes, axis=1)

    # Insert the time information as a new feature in the node values.
    node_values_np = np.insert(node_values_np, node_values_np.shape[-1],
                            values=time_of_the_day, axis=-1)
    
    # Add a dimension to the day of the week array.
    day_of_the_week = np.expand_dims(day_of_the_week, axis=-1)
    # Repeat for each node the time information at a specific timestamp.
    day_of_the_week = np.repeat(day_of_the_week, n_nodes, axis=1)

    # Insert the day of the week information as the last 7 features of the last axis
    node_values_np = np.insert(node_values_np, node_values_np.shape[-1],
                            values=day_of_the_week, axis=-1)

    return node_values_np

def _add_encoded_time_of_the_day(
    node_values_np: np.ndarray, node_values_df: pd.DataFrame
    ) -> np.ndarray:
    """
    Add a new feature to the node values numpy array containing
    the time of the day encoded as a value between 0 and 1.

    Parameters
    ----------
    node_values : ndarray
        The numpy array containing features of the nodes at different
        timestamps.
    node_values_df : DataFrame
        The dataframe containing the speed values of the nodes at
        different timestamps.

    Returns
    -------
    ndarray
        The numpy array containing the features of the nodes at
        different timestamps and the time of the day encoded as a value
        between 0 and 1.
    """
    # Encode the time of the day.
    time_of_the_day = np.array(
        [d.hour * 60 + d.minute 
         for d in pd.to_datetime(node_values_df.index.values)])

    # Get the minimum and maximum values of the selected criteria.
    min_value, max_value = (0, 23 * 60 + 59)
    # Scale the time information between 0 and 1.
    time_of_the_day = (time_of_the_day - min_value) / (max_value - min_value)

    # Get the number of nodes in the dataframe.
    _, n_nodes = node_values_df.shape

    # Add a dimension to the time of the day array.
    time_of_the_day = np.expand_dims(time_of_the_day, axis=-1)
    # Repeat for each node the time information at a specific timestamp.
    time_of_the_day = np.repeat(time_of_the_day, n_nodes, axis=1)

    # Insert the time information as a new feature in the node values.
    node_values_np = np.insert(node_values_np, node_values_np.shape[-1],
                            values=time_of_the_day, axis=-1)

    return node_values_np

def _add_encoded_day_of_the_week(
    node_values_np: np.ndarray, node_values_df: pd.DataFrame
    ) -> np.ndarray:
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
    day_of_the_week = np.array(node_values_df.index.day_of_week)

     # One hot encode the time information.
    day_of_the_week = np.eye(7)[day_of_the_week]

    # Get the number of nodes in the dataframe.
    _, n_nodes = node_values_df.shape

    # Add a dimension to the day of the week array.
    day_of_the_week = np.expand_dims(day_of_the_week, axis=1)

    # Repeat for each node the time information at a specific timestamp.
    day_of_the_week = np.repeat(day_of_the_week, n_nodes, axis=1)

    # Insert the day of the week information as the last 7 features of the last axis
    node_values_np = np.concatenate((node_values_np, day_of_the_week), axis=-1)

    #node_values_np = np.insert(node_values_np, node_values_np.shape[-1],
    #                           values=day_of_the_week, axis=-1)

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
    aggregate_by : 'time_of_day' ! 'day_of_week'
        The method by which the timestamps will be aggregated and then
        encoded.
        * 'time_of_day': encodes the timestamps by time of the day .
        * 'day_of_week': encodes the timestamps by day of the week [0-6]. 

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
        'day_of_week': node_values_df.index.day_of_week
    }

    # Get the time information by criteria.
    time_information = np.array(grouping_criteria[aggregate_by])

    if aggregate_by == 'time_of_day':
        # Get the minimum and maximum values of the selected criteria.
        min_value, max_value = (0, 23 * 60 + 59)
        # Scale the time information between 0 and 1.
        time_information = (time_information - min_value) / (max_value - min_value)

        # Get the number of nodes in the dataframe.
        _, n_nodes = node_values_df.shape

        # Add a dimension to the time information.
        time_information = np.expand_dims(time_information, axis=-1)
        # Repeat for each node the time information at a specific timestamp.
        time_information = np.repeat(time_information, n_nodes, axis=1)
    else:
        # One hot encode the time information.
        time_information = np.eye(7)[time_information]

        # Get the number of nodes in the dataframe.
        _, n_nodes = node_values_df.shape
        
        # Add a dimension to the time information.
        time_information = np.expand_dims(time_information, axis=1)
        
        # Repeat for each node the time information at a specific timestamp.
        time_information = np.repeat(time_information, n_nodes, axis=1)

    return time_information

def get_dataset_by_sliding_window(
    dataset: np.ndarray,
    time_dataset: np.ndarray,
    x_stepsize: int,
    y_stepsize: int
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
    time_dataset : ndarray
        The time information of the dataset.
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
    ndarray
        The time information of the first division of the dataset.
    ndarray
        The time information of the second division of the dataset.
    """
    # Initialize the dataset divisions as empty lists.
    x, x_time, y, y_time = [], [], [], []
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
        x_time.append(time_dataset[index : index + x_stepsize])
        y.append(dataset[index + x_stepsize : index + x_stepsize + y_stepsize])
        y_time.append(
            time_dataset[index + x_stepsize : index + x_stepsize + y_stepsize])

        # Increase the iteration count.
        i += 1

    # Stack the results.
    return np.stack(x), np.stack(y), np.stack(x_time), np.stack(y_time)

'''def get_distance_matrix(
    node_locations_df: pd.DataFrame,
    node_ids_dict: Dict[int, str]
    ) -> np.ndarray:
    # Build the distance matrix between the nodes.
    distance_matrix = np.zeros((len(node_ids_dict), len(node_ids_dict)))

    # Loop through each node.
    for id_i, idx_i in node_ids_dict.items():
        for id_j, idx_j in node_ids_dict.items():
            # Get the latitude and longitude of the two nodes.
            latitude_i = node_locations_df.loc[
                node_locations_df['sensor_id'] == id_i].latitude.values[0]
            longitude_i = node_locations_df.loc[
                node_locations_df['sensor_id'] == id_i].longitude.values[0]
            latitude_j = node_locations_df.loc[
                node_locations_df['sensor_id'] == id_j].latitude.values[0]
            longitude_j = node_locations_df.loc[
                node_locations_df['sensor_id'] == id_j].longitude.values[0]
            # Compute the haversine distance between the two nodes in miles
            # and assign it to the distance matrix.
            distance_matrix[idx_i, idx_j] = haversine(
                (latitude_i, longitude_i),
                (latitude_j, longitude_j)).miles

    return distance_matrix

def get_distance_matrix_1(
    node_distances_path: str,
    node_ids_dict: Dict[int, str],
    has_header: bool
    ) -> np.ndarray:
    if has_header:
        df = pd.read_csv(node_distances_path)
    else:
        df = pd.read_csv(node_distances_path, names=['from', 'to', 'cost'])
    
    # Build the distance matrix between the nodes.
    distance_matrix = np.zeros((len(node_ids_dict), len(node_ids_dict)))
    
    max_cost = df.cost.max()
    
    for id_i, idx_i in node_ids_dict.items():
        for id_j, idx_j in node_ids_dict.items():
            # Get the latitude and longitude of the two nodes.
            cost = df.loc[
                (df['from'] == int(id_i)) & (df['to'] == int(id_j))
                ]['cost'].values
            if not len(cost):
                cost = max_cost
            else:
                cost = cost[0]
            # Compute the haversine distance between the two nodes in miles
            # and assign it to the distance matrix.
            distance_matrix[idx_i, idx_j] = cost

    return distance_matrix'''

def get_distance_matrix(
    adj_matrix: np.ndarray,
    node_locations_df: pd.DataFrame,
    node_pos_dict: Dict[int, str]
    ) -> np.ndarray:
    # Get the distance matrix from the adjacency matrix.
    distance_matrix = 1 - adj_matrix
    # Set the distance between out-of-range nodes to 1_000.
    distance_matrix[distance_matrix == 1] = 1_000

    # Build the directed graph from the distance matrix.
    G = nx.DiGraph(distance_matrix)

    # Set the distance matrix between the nodes to be filled with miles.
    distance_matrix_in_miles = np.zeros((len(node_pos_dict), len(node_pos_dict)))

    # Loop through each node.
    for idx_i in node_pos_dict.keys():
        for idx_j in node_pos_dict.keys():
            # Get the shortest path between the two nodes.
            path = nx.dijkstra_path(G, idx_i, idx_j)
            distance = 0.
            for i in range(len(path) - 1):
                id_i = node_pos_dict[path[i]]
                id_j = node_pos_dict[path[i + 1]]
                # Get the latitude and longitude of the two nodes.
                latitude_i = node_locations_df.loc[
                    node_locations_df['sensor_id'] == id_i].latitude.values[0]
                longitude_i = node_locations_df.loc[
                    node_locations_df['sensor_id'] == id_i].longitude.values[0]
                latitude_j = node_locations_df.loc[
                    node_locations_df['sensor_id'] == id_j].latitude.values[0]
                longitude_j = node_locations_df.loc[
                    node_locations_df['sensor_id'] == id_j].longitude.values[0]

                distance += haversine(
                    (latitude_i, longitude_i),
                    (latitude_j, longitude_j)).miles
            # Compute the haversine distance between the two nodes in miles
            # and assign it to the distance matrix.
            distance_matrix_in_miles[idx_i, idx_j] = distance

    return distance_matrix_in_miles
