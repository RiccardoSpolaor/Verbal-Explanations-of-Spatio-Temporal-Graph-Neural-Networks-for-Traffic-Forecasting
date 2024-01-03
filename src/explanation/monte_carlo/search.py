"""
Module containing the Monte Carlo Tree Search algorithm used to find the
best input events subset that explains the selected target events in the
output spatial-temporal graph.
"""
import gc
import math
from typing import List, Tuple

import torch
import numpy as np

from .monte_carlo_tree import MonteCarloTreeSearch, Node
from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...data.data_processing import Scaler
from ...explanation.events import get_largest_event_set


def get_best_input_events_subset_by_mcts(
    x: np.ndarray,
    y: np.ndarray,
    distance_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    scaler: Scaler,
    n_rollouts: int,
    explanation_size: int,
    n_top_events: int,
    exploration_weight: float,
    remove_value: float = 0.,
    verbose: bool = False
    ) -> List[Tuple[int, int]]:
    """
    Get the best input events subset that explains the selected target
    events in the output spatial-temporal graph.

    Parameters
    ----------
    x : ndarray
        The input spatial-temporal graph.
    y : ndarray
        The output spatial-temporal graph.
    distance_matrix : ndarray
        The matrix containing the harvesine distance between each pair of
        nodes in miles.
    spatial_temporal_gnn : SpatialTemporalGNN
        The spatial-temporal graph neural network model used to predict the
        output spatial-temporal graph from the input spatial-temporal graph.
    scaler : Scaler
        The scaler used to scale the input and output spatial-temporal
        graphs.
    n_rollouts : int
        The number of rollouts to perform in the Monte Carlo Tree Search.
    explanation_size : int
        The explanation size, i.e. the maximum number of input events to
        be selected as an explanation of the target events.
    n_top_events : int
        The number of best input events with the best spatial-temporal plus
        speed correlation score with respect to the target events to be
        considered in the Monte Carlo Tree Search.
    exploration_weight : float
        The exploration weight used in the Monte Carlo Tree Search.
    remove_value : float, optional
        The value used to replace the features of the input spatial-temporal
        graph that are not related to the selected input events, by default 0.
    verbose : bool, optional
        Whether to print the execution information, by default False.

    Returns
    -------
    list of (int, int)
        The best input events subset that explains the selected target
        events in the output spatial-temporal graph.
    """
    # Copy the data in order to not modify the original.
    x = x.copy()
    y = y.copy()

    # Get the largest event set from the input data.
    input_events = get_largest_event_set(x)

    # Get the largest event set from the target data.
    target_events = get_largest_event_set(y)

    # Get the input events with their average spatio-temporal plus speed
    # correlation score with respect to the target events.
    input_events_with_scores = _get_input_events_with_score(
        x,
        y,
        input_events,
        target_events,
        distance_matrix)

    # Get the `top_n_input_events` input events.
    selected_input_events = [
        e for e, _ in input_events_with_scores[-n_top_events:] ]

    # Initialize the Monte Carlo Tree Search.
    mcts = MonteCarloTreeSearch(
        torch.FloatTensor(x),
        torch.FloatTensor(y),
        spatial_temporal_gnn,
        scaler,
        maximum_leaf_size=explanation_size,
        exploration_weight=exploration_weight,
        remove_value=remove_value)

    # Initialize the root node as the selected input events.
    root = Node(
        input_events=selected_input_events)

    # Loop over the number of rollouts.
    for i in range(n_rollouts):
        if verbose:
            print(f'Execution {i+1}/{n_rollouts}')
        # Perform a rollout from the root node.
        mcts.rollout(root)
        if verbose:
            print('reward:', mcts.best_leaf[1], ', mae:', mcts.best_leaf[2])
        if mcts.best_leaf[1] == float('inf'):
            break
        # Perform garbage collection.
        gc.collect()

    # Get the best input events subset.
    best_input_events_subset = mcts.get_best_input_events_subset()
    # Perform garbage collection.
    gc.collect()

    return best_input_events_subset

def _get_input_events_with_score(
    x: np.ndarray,
    y: np.ndarray,
    input_events: List[Tuple[int, int]],
    target_events: List[Tuple[int, int]],
    distance_matrix: np.ndarray
    ) -> List[Tuple[Tuple[int, int], float]]:
    """
    Get the input events with their average spatio-temporal plus speed
    correlation score with respect to the target events.

    Parameters
    ----------
    x : ndarray
        The input spatial-temporal graph.
    y : ndarray
        The output spatial-temporal graph.
    input_events : list of (int, int)
        The input events to be scored.
    target_events : list of (int, int)
        The target events with respect to which the input events are scored.
    distance_matrix : ndarray
        The matrix containing the harvesine distance between each pair of
        nodes in miles.

    Returns
    -------
    list of ( (int, int), float )
        The list of input events with their average spatio-temporal plus
        speed correlation score with respect to the target events.
    """
    # Set a list of input events with their average spatio-temporal plus
    # speed correlation score with respect to the target events.
    input_events_with_scores = []

    # Loop over the input events.
    for e in input_events:
        # Set the list of scores of the input event with respect to the
        # target events.
        event_scores = []

        # Get the set of all target node ids.
        target_nodes = np.unique([ n for _, n in target_events ])

        # Loop over the target nodes.
        for target_node in target_nodes:
            # Get all the timesteps at which the target node is active.
            target_timesteps = [ t for t, n in target_events
                                if n == target_node ]
            # Get the average speed of the target node among the timesteps.
            average_target_node_speed = np.mean(
                [ y[target_t, target_node] for target_t in target_timesteps ])
            # Get the last timestep at which the target node is active.
            last_target_timestep = np.max(target_timesteps)
            # Get the score of the input event with respect to the target node.
            score = _get_input_event_score(
                x,
                e[0],
                last_target_timestep,
                e[1],
                target_node,
                distance_matrix,
                average_target_node_speed)
            # Add the score to the list of scores of the input event.
            event_scores.append(score)

        # Get the average score of the input event with respect to the
        # target events.
        average_event_score = np.mean(event_scores).item()
        # Add the input event with its average score to the list of input
        # events with scores.
        input_events_with_scores.append((e, average_event_score))

    # Sort the input events with respect to their score.
    input_events_with_scores = sorted(
        input_events_with_scores, key=lambda ev: ev[1])
    return input_events_with_scores

def _get_input_event_score(
    x: np.ndarray,
    source_timestep: int,
    target_last_timestep: int,
    source_id: int,
    target_id: int,
    distance_matrix: np.ndarray,
    target_node_average_speed: float
    ) -> float:
    """
    Get the score of an input event with respect to a target event.
    The score is computed as the sum of the spatio-temporal correlation
    score and the speed correlation score.

    Parameters
    ----------
    x : ndarray
        The input spatial-temporal graph.
    source_timestep : int
        The timestep of the source node.
    target_last_timestep : int
        The last timestep of the target node.
    source_id : int
        The id of the source node.
    target_id : int
        The id of the target node.
    distance_matrix : ndarray
        The matrix containing the harvesine distance between each pair of
        nodes in miles.
    target_node_average_speed : float
        The average speed of the target node among the timesteps at which
        it is active.

    Returns
    -------
    float
        _description_
    """
    # Get the number of timesteps in the input data.
    n_x_timesteps = x.shape[0]
    # Get the speed of the source node.
    source_node_speed = x[source_timestep, source_id, 0]

    # If the average speed of the target node is zero,
    # then return a zero score.
    if target_node_average_speed == 0:
        return 0

    # Get the time difference between the source and target nodes in hours,
    # Knowing that each timestep is 5 minutes.
    delta_time = target_last_timestep + n_x_timesteps - source_timestep
    delta_time *= 5 / 60

    # Get the distance between the source and target nodes in miles.
    delta_distance = distance_matrix[source_id, target_id]
    # Get the absolute difference between the source node speed and
    # the target node average speed in miles/hour.
    delta_speed = abs(target_node_average_speed - source_node_speed)
    # Get the spatio-temporal correlation score.
    score = delta_time / (delta_distance + 1e-8)
    # Add the speed correlation score to the spatio-temporal correlation
    # score.
    score += (1 / (delta_speed + 1e-8)) * math.exp(
        -(delta_time + delta_distance))
    return score
