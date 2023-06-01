from typing import Tuple

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import numpy as np

from ...utils.config import MPH_TO_KMH_FACTOR


def get_adjacency_distance_matrix(
    adj_matrix: np.ndarray, n_timesteps: int) -> np.ndarray:
    """
    Build the adjacency distance matrix from the adjacency matrix.
    Values in the resulting matrix are in the range [0, 1], where 0
    means that the two nodes have no distance and 1 means that the two
    nodes have the maximum distance.
    The resulting matrix is a square matrix of size (N*T)x(N*T),
    where N is the number of nodes and T is the number of timesteps.

    Parameters
    ----------
    adj_matrix : ndarray
        The adjacency matrix.
    n_timesteps : int
        The number of timesteps.

    Returns
    -------
    ndarray
        The adjacency distance matrix covering the spatial-temporal
        graph at each timestep.
    """
    # Repeat the adjacency matrix n_timesteps times in both dimensions.
    adj_matrix_expanded = np.concatenate(
        [np.concatenate([adj_matrix] * n_timesteps, axis=0)] * n_timesteps,
        axis=1)

    # Compute the inverse of the adjacency matrix.
    adj_distance_matrix = 1 - adj_matrix_expanded
    return adj_distance_matrix

def get_temporal_distance_matrix(n_nodes: int, n_timesteps: int) -> np.ndarray:
    """
    Build the temporal distance matrix. Values in the resulting matrix
    are in the range [0, 1], where 0 means that the two nodes have no
    distance and 1 means that the two nodes have the maximum distance.
    The resulting matrix is a square matrix of size (N*T)x(N*T),
    where N is the number of nodes and T is the number of timesteps.

    Parameters
    ----------
    n_nodes : int
        The number of nodes to consider.
    n_timesteps : int
        The number of timesteps to consider.

    Returns
    -------
    ndarray
        The temporal distance matrix covering the spatial-temporal
        graph at each timestep.
    """
    # Line-space the time steps between 0 and 1.
    linespaced_time_steps = np.linspace(0, 1, n_timesteps)

    # Repeat each time step for each node.
    extended_time_steps = np.repeat(linespaced_time_steps, n_nodes)

    # Add dummy dimension to the array.
    extended_time_steps = np.expand_dims(extended_time_steps, axis=1)

    # Compute the time distance matrix.
    time_distance_matrix = cdist(extended_time_steps, extended_time_steps,
                                 'euclidean')
    return time_distance_matrix

from sklearn.cluster import DBSCAN

def get_clusters(
    instance: np.ndarray, adj_distance_matrix: np.ndarray,
    temporal_distance_matrix: np.ndarray, eps: float,
    min_samples: int, speed_distance_weight: float = 3) -> np.ndarray:
    """
    Get the clusters of the given instance using the DBSCAN algorithm.

    Parameters
    ----------
    instance : ndarray
        The spatial-temporal graph instance to cluster.
    adj_distance_matrix : ndarray
        The adjacency matrix of the nodes in the graph measured in distance
        between 0 and 1.
    temporal_distance_matrix : ndarray
        The matrix measuring the distance between the time steps
        of the nodes in the graph between 0 and 1.
    eps : float
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int
        The number of samples in a neighborhood for a point for it to be
        considered as a core point.
    speed_distance_weight : float, optional
        The weight of the speed distance in the clustering process,
        by default 3.

    Returns
    -------
    ndarray
        The clusters of the given instance.
    """
    n_timesteps, n_nodes, _ = instance.shape

    # Reshape the instance to be a column vector.
    instance = instance.reshape(-2, 1)

    # Compute the distance matrix between the speed of the nodes in the graph.
    speed_distance_matrix = cdist(instance, instance, 'euclidean')
    # Normalize the distance matrix between 0 and 1.
    speed_distance_matrix /= np.max(speed_distance_matrix)

    # Compute the weighted distance matrix between the nodes in the graph
    # in terms of speed and the spatial and temporal distances.
    distance_matrix = speed_distance_matrix * speed_distance_weight +\
        adj_distance_matrix + temporal_distance_matrix

    # Normalize the distance matrix between 0 and 1.
    #distance_matrix /= np.max(distance_matrix)
    # Set the distance between nodes that are not connected to an
    # unreachable value.
    distance_matrix[adj_distance_matrix == 1] = 1_000

    # Compute the clusters of the given instance using the DBSCAN algorithm.
    dbscan = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples,
                    n_jobs=-1)
    clusters = dbscan.fit_predict(distance_matrix)

    # Add a dummy dimension to the clusters array.
    clusters = np.expand_dims(clusters, axis=1)

    # Reshape the clusters array to have the same shape as the instance.
    clusters = clusters.reshape(n_timesteps, n_nodes, 1)

    return clusters

def get_dataset_for_explainability(
    x: np.ndarray, y: np.ndarray, eps: float, min_samples: int,
    adj_distance_matrix: np.ndarray, temporal_distance_matrix: np.ndarray,
    congestion_max_speed: float = 60, free_flow_min_speed: float = 110,
    is_x_kmph: bool = False, is_y_kmph: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dataset which target predictions are meaningful for
    explainability.
    The function iterates through a dataset inputs and its corresponding
    predicted target values. It applies clustering on the target values in
    order to find clusters of similar speed. Then it selects the clusters
    with average speed values corresponding to congestions or free-flow
    traffic. For each of these clusters, it associates the corresponding
    input generating new dataset instances.

    Parameters
    ----------
    x : ndarray
        The input dataset.
    y : ndarray
        The predicted target dataset.
    eps : float
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int
        The number of samples (or total weight) in a neighborhood for a
        point to be considered as a core point. This includes the point
        itself.
    adj_distance_matrix : ndarray
        The adjacency distance matrix of the spatial-temporal graphs
        instances of the dataset.
    temporal_distance_matrix : ndarray
        The temporal distance matrix of the spatial-temporal graph
        instances of the dataset.
    congestion_max_speed : float, optional
        The maximum speed value for a cluster to be considered as a
        congestion, by default 60.
    free_flow_min_speed : float, optional
        The minimum speed value for a cluster to be considered as a
        free-flow traffic, by default 110.
    is_kmph : bool, optional
        Whether the speed values are in km/h or miles/h, by default True.
        Note that the resulting speeds will be always converted to
        miles/h.

    Returns
    -------
    ndarray
        The resulting input dataset for explainability.
    ndarray
        The respective target dataset for explainability.
    """
    # Initialize the resulting datasets.
    x_for_explainability, y_for_explainability = [], []

    # Iterate through the dataset instances.
    for x_, y_ in zip(x[:100], y[:100]):
        # Get the clusters of the target values.
        clusters = get_clusters(
            y_, adj_distance_matrix, temporal_distance_matrix, eps,
            min_samples)
        # Iterate through the clusters.
        for c in np.unique(clusters):
            # Ignore the noise points.
            if c == -1:
                continue
            # Mask the target values according to the cluster.
            masked_y_ = y_ * (clusters == c)
            # Get the mean of the target values in the cluster.
            cluster_nodes_mean = masked_y_[masked_y_ > 0]
            
            # If the cluster mean is in the range of congestion or free-flow
            # traffic, generate a new instance.
            if np.all(cluster_nodes_mean <= congestion_max_speed) or \
                np.all(cluster_nodes_mean >= free_flow_min_speed):
                x_for_explainability.append(x_)
                y_for_explainability.append(masked_y_)
    # Turn the resulting datasets into numpy arrays.
    x_for_explainability = np.array(x_for_explainability)
    y_for_explainability = np.array(y_for_explainability)
    
    # Convert the speed values to miles/h if needed.
    if is_x_kmph:
        x_for_explainability = x_for_explainability / MPH_TO_KMH_FACTOR
    if is_y_kmph:
        y_for_explainability = y_for_explainability / MPH_TO_KMH_FACTOR

    return x_for_explainability, y_for_explainability
