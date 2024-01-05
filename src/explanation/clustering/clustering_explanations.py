from typing import List

from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from .evaluation import get_connected_cluster_dissimilarity, get_within_clusters_variance


def get_explanation_clusters(
    x: np.ndarray,
    adj_distance_matrix: np.ndarray,
    temporal_distance_matrix: np.ndarray,
    speed_distance_weight: float = 3,
    n_clusters_list: List[int] = [i for i in range(1, 6)],
    ) -> np.ndarray:
    """
    Get the clusters of the given explanation instance using the
    k-medoids algorithm.

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
    speed_distance_weight : float, optional
        The weight of the speed distance in the clustering process,
        by default 3.
    n_clusters : int, optional
        The number of clusters to find, by default 4.

    Returns
    -------
    ndarray
        The clusters of the given instance.
    """
    n_timesteps, n_nodes, _ = x.shape

    # Reshape the instance to be a column vector.
    reshaped_instance = x.reshape(-2, 1)

    # Compute the distance matrix between the speed of the nodes in the graph.
    speed_distance_matrix = cdist(
        reshaped_instance,
        reshaped_instance,
        'euclidean')
    # Normalize the distance matrix between 0 and 1.
    speed_distance_matrix /= np.max(speed_distance_matrix)

    # Compute the weighted distance matrix between the nodes in the graph
    # in terms of speed and the spatial and temporal distances.
    distance_matrix = speed_distance_matrix * speed_distance_weight +\
        adj_distance_matrix + temporal_distance_matrix

    # Set the distance between nodes that are not connected to an
    # unreachable value.
    distance_matrix[adj_distance_matrix == 1] = 1_000

    # Get the non-zero indices of the reshaped instance.
    non_zeros = np.where(reshaped_instance != 0)[0]
    # Reduce distance matrix by solely considering the nodes that are
    # present in the instance.
    distance_matrix = distance_matrix[non_zeros, :][:, non_zeros]
    
    if not len(distance_matrix):
        return np.full_like(reshaped_instance, -1).reshape(n_timesteps, n_nodes, 1).astype(int)

    best_clusters = None
    best_dissimilarity_score = float('-inf')
    
    for n_clusters in n_clusters_list:
        # Compute the clusters of the given instance using the k-medoids
        # algorithm.
        agglomerative_clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='single')

        clusters = agglomerative_clustering.fit_predict(distance_matrix)

        # Add a dummy dimension to the clusters array.
        clusters = np.expand_dims(clusters, axis=1)

        # Create a cluster vector with dummy -1 values.
        clusters_vector = np.full_like(reshaped_instance, -1)
        # Set the non-zero values of the cluster vector to the clusters
        clusters_vector[non_zeros] = clusters[:]

        # Reshape the clusters array to have the same shape as the instance.
        clusters_vector = clusters_vector.reshape(n_timesteps, n_nodes, 1)

        # Set the cluster IDs as integers.
        clusters_vector = clusters_vector.astype(int)

        # Compute the dissimilarity score of the clusters.
        dissimilarity_score = get_connected_cluster_dissimilarity(
            x, clusters_vector, ignore_noise=True)

        # Tweak it by the within clusters variance.
        dissimilarity_score /= get_within_clusters_variance(
            x, clusters_vector, ignore_noise=True)

        # Update the best clusters.
        if dissimilarity_score > best_dissimilarity_score:
            best_clusters = clusters_vector
            best_dissimilarity_score = dissimilarity_score

    return best_clusters

def get_explanation_clusters_dataset(
    x: np.ndarray,
    adj_distance_matrix: np.ndarray,
    temporal_distance_matrix: np.ndarray) -> np.ndarray:
    """
    Get the clusters of the given explanation instances using the
    Agglomerative Clustering algorithm.
    
    Parameters
    ----------
    x : ndarray
        The explanation dataset.
    adj_distance_matrix : ndarray
        The adjacency distance matrix of the spatial-temporal graphs
        instances of the dataset.
    temporal_distance_matrix : ndarray
        The temporal distance matrix of the spatial-temporal graph
        instances of the dataset.
    
    Returns
    -------
    ndarray
        The clusters of the given explanation instances.
    """
    # Initialize the resulting datasets.
    clusters_list = []

    # Iterate through the dataset instances.
    for x_ in x:
        # Get the clusters of the input values.
        clusters_list.append(get_explanation_clusters(
            x_, adj_distance_matrix, temporal_distance_matrix))

    return np.array(clusters_list)
