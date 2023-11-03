import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.config import MPH_TO_KMH_FACTOR


def get_node_values_dataframe(
    hdf_file_path: str, turn_in_kmph: bool = True) -> pd.DataFrame:
    """
    Get the dataframe containing the speed values of the nodes
    at different timestamps from a specific `hdf` file.
    The returning dataframe will have the different timestamps as
    indices and each column refers to a specific node. The cells
    contain speed information for the node at a timestamp.

    Parameters
    ----------
    hdf_file_path : str
        The path of the `hdf` file from which the dataframe is built.
    turn_in_kmph : bool, optional
        Whether to turn the speed values of the dataframe to
        km/h or to leave them in miles/hour, by default True.

    Returns
    -------
    DataFrame
        The dataframe containing the speed values of the nodes at different
        timestamps.
    """
    # Get the dataframe containing the node values at different timestamps.
    node_values_df = pd.read_hdf(hdf_file_path)

    if turn_in_kmph:
        # Transform the speed from miles/h to km/h.
        node_values_df.iloc[:] = node_values_df.iloc[:] * MPH_TO_KMH_FACTOR

    # Assign the column type as str.
    node_values_df.columns = [str(c) for c in node_values_df.columns]

    return node_values_df

def get_adjacency_matrix(
    adj_matrix_file_path: str
    ) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    """
    Get the adjacency matrix structured data from a pickle file.
    The adjacency matrix is a (N, N) shaped matrix, where N is
    the number of nodes. Each cell contains the spatial distance
    information among the two considered nodes.

    Parameters
    ----------
    adj_matrix_file_path : str
        The path where the pickle file containing the adjacency
        matrix structured data is located.

    Returns
    -------
    list of str
        The header of the adjacency matrix specifying in order to which
        node each column (and row) refers.
    { str: int }
        Dictionary specifying for each node id the corresponding index
        in the matrix columns (and rows).
    ndarray
        The adjacency matrix that contains the spatial distance among
        each node.
    """
    # Load the adjacency matrix from the pickle file.
    with open(adj_matrix_file_path, 'rb') as f:
        adj_matrix_structure = pickle.load(f, encoding='latin1')

    return adj_matrix_structure

def get_locations_dataframe(
    hdf_file_path: str,
    has_header: bool
    ) -> pd.DataFrame:
    # Get the dataframe containing the latitude and longitude of each sensor.
    if has_header:
        locations_df = pd.read_csv(hdf_file_path, index_col='index')
    else:
        # Manually assign the header and the index name.
        locations_df = pd.read_csv(
            hdf_file_path, names=['sensor_id', 'latitude', 'longitude'])
        locations_df.index.name = 'index'

    # Set the sensor_id feature as string values.
    locations_df['sensor_id'] = locations_df['sensor_id'].apply(str)
    return locations_df
