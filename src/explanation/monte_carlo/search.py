import gc
import math
from typing import List, Optional, Tuple

import torch
import numpy as np

from .monte_carlo_tree import MonteCarloTreeSearch, Node
from ..events import remove_features_by_events
from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...data.data_processing import Scaler
from ...explanation.events import get_largest_event_set
from ...explanation.navigator.model import Navigator


def get_best_input_events_subset(
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
    # Copy the data in order to not modify the original.
    x = x.copy()
    y = y.copy()
    
    # Get the largest event set from the input data.
    input_events = get_largest_event_set(x)

    # Get the largest event set from the target data.
    target_events = get_largest_event_set(y)

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

def get_explanations_from_data(
    x: np.ndarray,
    y: np.ndarray,
    adj_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    navigator: Navigator,
    distance_matrix: np.ndarray,
    scaler: Scaler,
    n_rollouts: int = 20,
    explanation_size: Optional[int] = None,
    exploration_weight: int = 500,
    top_n_input_events: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

    explained_x, explained_y = [], []

    for x_, y_ in zip(x, y):
        if explanation_size is None:
            explanation_size_ = int((y_.flatten() != 0).sum() * 1.5)
        else:
            explanation_size_ = explanation_size
        if top_n_input_events is None:
            top_n_input_events_ = explanation_size_ * 2
        else:
            top_n_input_events_ = top_n_input_events
        subset = get_best_input_events_subset(
            x_,
            y_,
            distance_matrix,
            spatial_temporal_gnn,
            scaler,
            n_rollouts=n_rollouts,
            explanation_size=explanation_size_,
            n_top_events=top_n_input_events_,
            exploration_weight=exploration_weight)

        # TODO: Fix this to add other events.
        input_events_subset = [ ( 0, e[0], e[1] ) for e in subset[0].input_events ]

        x_ = remove_features_by_events(x_, input_events_subset)

        x_ = x_[..., :1]

        explained_x.append(x_)
        explained_y.append(y_)

    return np.array(explained_x), np.array(explained_y)

# MPH_TO_KMH_FACTOR = 1.609344
#CONGESTION_SPEED = 60 # mph #/ MPH_TO_KMH_FACTOR

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
    distance_matrix : np.ndarray
        _description_
    target_node_average_speed : float
        _description_

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
    delta_time = (target_last_timestep + n_x_timesteps - source_timestep) 
    delta_time *= 5 / 60

    # Get the distance between the source and target nodes in miles.
    delta_distance = distance_matrix[source_id, target_id]
    # Get the absolute difference between the source node speed and
    # the target node average speed in miles/hour.
    delta_speed = abs(target_node_average_speed - source_node_speed)
    # Get the spatio-temporal correlation score.
    score = (delta_time / (delta_distance + 1e-8))
    # Add the speed correlation score to the spatio-temporal correlation score.
    score += (1 / delta_speed + 1e-8) * math.exp(-(delta_time + delta_distance))
    return score
    '''#return  C + (1 - abs(speed_source - CONGESTION_SPEED) / CONGESTION_SPEED) * (1 - abs(speed_target - CONGESTION_SPEED) / CONGESTION_SPEED)
    if speed_target >= CONGESTION_SPEED:
        return  C + math.log( source_node_speed + 1, CONGESTION_SPEED + 1 ) * math.exp(-(delta_time + delta_distance)) #math.log(speed_target.item(), CONGESTION_SPEED)
    else:
        return C + (1 / math.log( source_node_speed + 1, CONGESTION_SPEED + 1 )) * math.exp(-(delta_time + delta_distance)) #math.log(speed_target.item(), CONGESTION_SPEED)
    # If I don't have a target congestion
    # - I want to deprioritize nodes that are expected to lead to congestion (C > 1)
    # - I want to prioritize nodes that are expected to lead to no congestion (C < 1)
    # - I want to prioritize lower speeds, since it is easier to create a free flow of traffic if the traffic is flowing slower.
    if speed_target.item() >= CONGESTION_SPEED:
        return  C * (CONGESTION_SPEED / speed_target.item() )#math.log(speed_target.item(), CONGESTION_SPEED)
    # If I have a target congestion
    # - I want to deprioritize nodes that are expected to lead to no congestion (C < 1)
    # - I want to prioritize nodes that are expected to lead to congestion (C > 1)
    # - I want to prioritize higher speeds, since it is easier to get to congestion if the traffic is flowing faster.
    else:
        return C * (speed_target.item() / CONGESTION_SPEED)#math.log(speed_target.item(), CONGESTION_SPEED)

    #return (math.log(speed_target.item(), CONGESTION_SPEED) - 1) * (speed_source * delta_time / (delta_distance + 1e-1) - 1)

    #return (math.log(speed_target, CONGESTION_SPEED) - 1) * (speed_source * delta_time / (delta_distance + 1e-1) - 1)'''