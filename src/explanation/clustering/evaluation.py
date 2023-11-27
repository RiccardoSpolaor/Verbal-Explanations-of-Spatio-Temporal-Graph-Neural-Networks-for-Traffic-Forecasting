from time import time
from typing import List, Tuple

from sklearn.model_selection import ParameterGrid
import numpy as np

from .clustering import get_clusters, get_explanation_clusters
from .metrics import (
    get_within_clusters_variance,
    get_connected_cluster_dissimilarity,
    get_noise_ratio)


def apply_grid_search_on_explanation_dataset(
    x: np.ndarray,
    adj_distance_matrix: np.ndarray,
    temporal_distance_matrix: np.ndarray,
    speed_distance_weight_list: List[float],
    n_clusters_list: List[int],
    ) -> None:
    """
    Apply grid search on the given explanation dataset. The clustering scores
    are the average within-cluster variance, the average connected cluster
    dissimilarity, and the average noise ratio. The clustering function used
    is k-medoids.

    Parameters
    ----------
    x : ndarray
        The spatial-temporal explanation instances to be clustered and
    adj_distance_matrix : ndarray
        The adjacency distance matrix of the spatial-temporal graphs.
    temporal_distance_matrix : ndarray
        The temporal distance matrix of the spatial-temporal graphs.
    speed_distance_weight_list : list of float
        The weights of the speed distance in the distance matrix to
        be tested.
    n_clusters_list : list of int
        The number of clusters to form as well as the number of centroids to
        generate to be tested.
    """
    parameter_grid = ParameterGrid({
        'speed_distance_weight': speed_distance_weight_list,
        'n_clusters': n_clusters_list })

    for p in parameter_grid:
        print('Testing:', *[f'{k}: {v}' for k, v in p.items()])
        get_explanation_dataset_clustering_scores(
             x,
             adj_distance_matrix,
             temporal_distance_matrix,
             **p)

def get_explanation_dataset_clustering_scores(
    x: np.ndarray,
    adj_distance_matrix: np.ndarray,
    temporal_distance_matrix: np.ndarray,
    speed_distance_weight: int,
    n_clusters: int,
    ) -> None:
    """
    Get the clustering scores applied on the given explanation dataset.
    The clustering scores are the average within-cluster variance,
    the average connected cluster dissimilarity, and the average noise ratio.
    The clustering function used is k-medoids.

    Parameters
    ----------
    x : ndarray
        The spatial-temporal explanation instances to be clustered and
        evaluated.
    adj_distance_matrix : ndarray
        The adjacency distance matrix of the spatial-temporal graphs.
    temporal_distance_matrix : ndarray
        The temporal distance matrix of the spatial-temporal graphs.
    speed_distance_weight : float
        The weight of the speed distance in the distance matrix.
    n_clusters : int
        The number of clusters to form as well as the number of centroids to
        generate.

    """
    # Initialize the running total metrics.
    running_within_cluster_variance = 0.
    running_connected_cluster_dissimilarity = 0.
    running_time = 0.

    beginning_time = time()

    # Filter out any instance with all zeros.
    x = x[np.any(x != 0, axis=(1, 2, 3))]

    for i, x_ in enumerate(x):
        start_time = time()
        steps = i + 1
        # Get the clusters for the current instance.
        clusters = get_explanation_clusters(
            x_,
            adj_distance_matrix,
            temporal_distance_matrix,
            speed_distance_weight,
            n_clusters)

        running_time += time() - start_time

        # Compute the metrics for the current instance.
        within_cluster_variance = get_within_clusters_variance(
            x_,
            clusters,
            ignore_noise=True)

        connected_cluster_dissimilarity = get_connected_cluster_dissimilarity(
            x_,
            clusters,
            ignore_noise=True)

        # Update the running total metrics.
        running_within_cluster_variance += within_cluster_variance
        running_connected_cluster_dissimilarity += connected_cluster_dissimilarity
        

        print(
            f'[{steps}/{len(x)}] -',
            f'{time() - beginning_time:.0f}s -',

            f'Within Cluster Variance: {running_within_cluster_variance / steps:.3g} -',
            f'Connected Clusters Dissimilarity: {running_connected_cluster_dissimilarity / steps:.3g} -',
            f'Average time: {running_time / steps:.3g}s',

            '             ' if steps < len(x) else '',
            end='\r' if steps < len(x) else '\n')

def apply_grid_search(
    instances: np.ndarray,
    eps_list: List[float],
    min_samples_list: List[int],
    adj_distance_matrix: np.ndarray,
    temporal_distance_matrix: np.ndarray
    ) -> None:
    parameter_grid = ParameterGrid({
        'eps': eps_list, 'min_samples': min_samples_list })

    for p in parameter_grid:
        print('eps:', p['eps'], 'min_samples:', p['min_samples'])

        (avg_within_cluster_variance, avg_connected_cluster_dissimilarity,
         avg_noise_ratio) = get_explained_dataset_clustering_scores(
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
