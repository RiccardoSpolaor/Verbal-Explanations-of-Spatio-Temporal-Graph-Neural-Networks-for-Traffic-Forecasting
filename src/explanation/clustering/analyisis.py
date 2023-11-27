from typing import Dict, Optional
import numpy as np
import pandas as pd


def get_node_values_with_clusters_and_location_dataframe(
    node_values: np.ndarray,
    clusters: np.ndarray,
    node_pos_dict: Dict[int, str],
    locations_df: pd.DataFrame,
    time_values: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
    """
    Get a pandas dataframe from a numpy array of node speed values a numpy
    array of node clusters and a pandas dataframe of node locations.
    The resulting dataframe has for each timestamp the value of the speed
    for each node, its cluster and the location of the node in the form of
    latitude and longitude.

    Parameters
    ----------
    node_values : ndarray
        The numpy array containing the values of the speed of each node
        for each timestamp.
    clusters : ndarray
        The numpy array containing the cluster of each node for each
        timestamp.
    node_pos_dict : { int: str }
        The dictionary containing the position of each node and the
        corresponding node ID.
    locations_df : DataFrame
        The dataframe containing the location of each node.
    time_values : ndarray, optional
        The numpy array containing the timestamps of the speed values.

    Returns
    -------
    DataFrame
        The resulting dataframe containing the values of the speed for
        each node, its cluster and the location of the node in the form of
        latitude and longitude.
    """
    # Concatenate the speeds and the cluster information.
    node_values = np.concatenate((node_values, clusters), axis=2)
    nodes_information = []

    for time_idx, node_matrix in enumerate(node_values):
        for node_idx, (speed, cluster) in enumerate(node_matrix):
            if time_values is not None:
                # Get the timestamp from the time values array.
                datetime = time_values[time_idx, node_idx]
            else:
                datetime = time_idx
            # Get the node ID from the node position dictionary.
            node_id = node_pos_dict[node_idx]
            # Get the latitude and longitude of the node.
            latitude = locations_df.loc[
                locations_df['sensor_id'] == node_id].latitude.values[0]
            longitude = locations_df.loc[
                locations_df['sensor_id'] == node_id].longitude.values[0]

            # Update the nodes information list.
            nodes_information.append(
                [node_id, latitude, longitude, cluster, speed, datetime])

    # Build the dataframe from the nodes information list.
    df = pd.DataFrame({
        'sensor_id': [n[0] for n in nodes_information],
        'latitude': [n[1] for n in nodes_information],
        'longitude': [n[2] for n in nodes_information],
        'cluster': [n[3] for n in nodes_information],
        'speed': [n[4] for n in nodes_information],
        'datetime': [n[5] for n in nodes_information]
    })

    # Set the cluster column as integer.
    try:
        df['cluster'] = df['cluster'].astype(int)
    except ValueError:
        pass

    return df
