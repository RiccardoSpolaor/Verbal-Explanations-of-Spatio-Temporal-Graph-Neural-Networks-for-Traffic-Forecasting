from typing import List, Tuple

from sklearn.model_selection import ParameterGrid
import numpy as np

from .clustering import get_clusters
from .metrics import (
    get_within_clusters_variance,
    get_connected_cluster_dissimilarity,
    get_noise_ratio)


def apply_grid_search(
    instances: np.ndarray,
    eps_list: List[float],
    min_samples_list: List[int],
    adj_distance_matrix: np.ndarray,
    temporal_distance_matrix: np.ndarray
    ) -> None:
    """
    Apply the grid search on the clustering of the predicted events for a
    given dataset.

    Parameters
    ----------
    instances : ndarray
        The spatial-temporal instances to be clustered and evaluated.
    eps_list : list of float
        The eps hyperparameter values to be tested.
    min_samples_list : list of int
        The min_samples hyperparameter values to be tested.
    adj_distance_matrix : ndarray
        The spatial distance matrix of the spatial-temporal graphs.
    temporal_distance_matrix : ndarray
        The temporal distance matrix of the spatial-temporal graphs.
    """
    parameter_grid = ParameterGrid({
        'eps': eps_list, 'min_samples': min_samples_list })

    for p in parameter_grid:
        print('eps:', p['eps'], 'min_samples:', p['min_samples'])

        (avg_within_cluster_variance, avg_connected_cluster_dissimilarity,
         avg_noise_ratio) = get_dataset_clustering_scores(
             instances, adj_distance_matrix, temporal_distance_matrix,
             p['eps'], p['min_samples'])

        print(
            '\tWithin-Cluster Variance:',
            f'{avg_within_cluster_variance:.3g}',
            'Connected Cluster Dissimilarity:',
            f'{avg_connected_cluster_dissimilarity:.3g}',
            'Noise points ratio:', f'{avg_noise_ratio:.3g}',
            end='\n\n')

def get_dataset_clustering_scores(
    instances: np.ndarray, adj_distance_matrix: np.ndarray,
    temporal_distance_matrix: np.ndarray, eps: float, min_samples: int
    ) -> Tuple[float, float, float]:
    """
    Get the clustering scores applied on the given dataset.
    The clustering scores are the average within-cluster variance,
    the average connected cluster dissimilarity, and the average noise ratio.

    Parameters
    ----------
    instances : ndarray
        The spatial-temporal instances to be clustered and evaluated.
    adj_distance_matrix : ndarray
        The adjacency distance matrix of the spatial-temporal graphs.
    temporal_distance_matrix : ndarray
        The temporal distance matrix of the spatial-temporal graphs.
    eps : float
        The maximum distance between two samples for them to be considered as
        in the same neighborhood.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered
        as a core point.

    Returns
    -------
    float
        The average within-cluster variance.
    float
        The average connected cluster dissimilarity.
    float
        The average noise ratio.
    """
    # Initialize the running total metrics.
    total_within_cluster_variance = 0.
    total_connected_cluster_dissimilarity = 0.
    total_noise_ratio = 0.

    for instance in instances:
        # Get the clusters for the current instance.
        clusters = get_clusters(
            instance, adj_distance_matrix, temporal_distance_matrix,
            eps=eps, min_samples=min_samples)

        # Compute the metrics for the current instance.
        within_cluster_variance = get_within_clusters_variance(
            instance, clusters)
        connected_cluster_dissimilarity = \
            get_connected_cluster_dissimilarity(instance, clusters)
        noise_ratio = get_noise_ratio(instance, clusters)

        # Update the running total metrics.
        total_within_cluster_variance += within_cluster_variance
        total_connected_cluster_dissimilarity += \
            connected_cluster_dissimilarity
        total_noise_ratio += noise_ratio

    # Get the average metrics.
    avg_within_cluster_variance = \
        total_within_cluster_variance / len(instances)
    avg_connected_cluster_dissimilarity = \
        total_connected_cluster_dissimilarity / len(instances)
    avg_noise_ratio = total_noise_ratio / len(instances)

    return (avg_within_cluster_variance, avg_connected_cluster_dissimilarity,
            avg_noise_ratio)
