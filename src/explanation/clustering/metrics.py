import numpy as np


def get_within_clusters_variance(
    instance: np.ndarray, clusters: np.ndarray) -> float:
    """
    Get the Within-Cluster Variance metric of the clusters
    obtained on the given instance in terms of speed.

    Parameters
    ----------
    instance : ndarray
        The spatial-temporal graph instance on which the clusters
        are evaluated.
    clusters : ndarray
        The clusters obtained on the given instance.

    Returns
    -------
    float
        The Within-Cluster Variance metric result.
    """
    # Set the intial value of the numerator sum to 0.
    numerator_sum = 0.
    # Set the initial value of the total number of nodes to 0.
    total_node_number = 0.

    for c in np.unique(clusters):
        # Get the sub-sample of the nodes in the graph that belong to the
        # current cluster.
        sub_sample = instance[clusters == c]
        # Get the length of the sub-sample.
        len_sub_sample = len(sub_sample)
        # Update the total nominator sum.
        numerator_sum += np.var(sub_sample) * len_sub_sample
        # Update the total number of nodes with the length of the sub-sample.
        total_node_number += len_sub_sample

    return numerator_sum / (total_node_number * np.var(instance))

def get_connected_cluster_dissimilarity(
    instance: np.ndarray, clusters: np.ndarray) -> float:
    """
    Get the Connected Cluster Dissimilarity metric of the clusters
    obtained on the given instance in terms of speed.

    Parameters
    ----------
    instance : ndarray
        The spatial-temporal graph instance on which the clusters
        are evaluated.
    clusters : ndarray
        The clusters obtained on the given instance.

    Returns
    -------
    float
        The Connected Cluster Dissimilarity metric result.
    """
    # Get the total unique cluster IDs.
    total_clusters = np.unique(clusters)

    # Set the initial value of the denominator sum to 0.
    denominator_sum = 0.
    # Set the initial value of the nominator sum to 0.
    nominator_sum = 0.

    for i, c1 in enumerate(total_clusters):
        for c2 in total_clusters[i+1:]:
            # If the two clusters are not connected, continue the loop.
            #if not are_clusters_connected(clusters, c1, c2, adj_matrix):
            #    continue
            # Get the sub-samples of the nodes in the graph that belong to the
            # current clusters.
            sub_sample1 = instance[clusters == c1]
            sub_sample2 = instance[clusters == c2]
            # Get the length of the sub-samples.
            len_sub_sample1 = len(sub_sample1)
            len_sub_sample2 = len(sub_sample2)
            # Compute the square root of the product of the lengths.
            sqrt_lens = np.sqrt(len_sub_sample1 * len_sub_sample2)
            # Compute the absolute difference between the means.
            abs_mean_diff = np.abs(np.mean(sub_sample1) - np.mean(sub_sample2))
            # Update the nominator sum.
            nominator_sum += sqrt_lens * abs_mean_diff
            # Update the denominator sum.
            denominator_sum += sqrt_lens

    return nominator_sum / denominator_sum if denominator_sum > 0 else 0

def get_noise_ratio(instance: np.ndarray, clusters: np.ndarray) -> float:
    """
    Get the Noise Ratio metric of the clusters obtained on the given
    instance.

    Parameters
    ----------
    instance : ndarray
        The spatial-temporal graph instance on which the clusters
        are evaluated.
    clusters : ndarray
        The clusters obtained on the given instance.

    Returns
    -------
    float
        The Noise Ratio metric result.
    """
    noise = clusters[clusters == -1]
    return len(noise) / len(instance.flatten())