from typing import List, Optional, Tuple
import torch
import numpy as np

from ..events import remove_features_by_events
from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...data.data_processing import Scaler
from ...explanation.events import get_largest_event_set
from ...explanation.navigator.model import Navigator
from .monte_carlo_tree import MonteCarloTreeSearch, Node


def get_best_input_subset(
    x: torch.FloatTensor,
    y: torch.FloatTensor,
    adj_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    navigator: Navigator,
    scaler: Scaler,
    n_rollouts: int = 50,
    maximum_leaf_size: int = 50,
    exploration_weight: float = 500,
    top_n_input_events: int = 500,
    verbose: bool = False
    ) -> List[List[int]]:
    # Copy the data in order to not modify the original.
    x = x.copy()
    y = y.copy()
    
    # Get the largest event set from the input data.
    input_events = get_largest_event_set(x)
    input_events = [i for i in input_events if i[0] == 0]

    # Get the largest event set from the output data.
    target_events = [
        [e[1], e[2], y[e[1], e[2], 0]] for e in get_largest_event_set(y)]

    # Set the input events set along with the correlation score to
    # the target events.
    input_events_with_correlation_score = []

    for i, e in enumerate(input_events):
        # Get the encoded information of the input event with respect to the
        # input data.
        encoded_information = x[e[1], e[2], :]
        # Set the enoded input event.
        e_ = (e[1], e[2], *encoded_information)
        
        # Set a batch of target events.
        target_batch = torch.FloatTensor(target_events).to(navigator.device)
        
        # Add a batch dimension to the input event.
        e_unsqueezed = np.expand_dims(e_, axis=0)
        # Set a batch of repeated input event of the same size as the target
        # events batch.
        input_batch = torch.FloatTensor(
            np.repeat(e_unsqueezed, len(target_events), axis=0)
            ).to(navigator.device)

        # Get the average correlation score average between the input event
        # and each target event.
        correlation_scores = navigator(input_batch, target_batch)
        correlation_score_avg = correlation_scores.mean().item()
        # Add the input event along with the correlation score to the list.
        input_events_with_correlation_score.append((e_, correlation_score_avg))

    # Sort the input events with respect to the correlation score.
    input_events_with_correlation_score = sorted(
        input_events_with_correlation_score, key=lambda x: x[1])
    # Get the `top_n_input_events` input events.
    input_events_with_correlation_score = input_events_with_correlation_score[-top_n_input_events:]

    # Get the number of timesteps.
    n_timesteps = x.shape[-3]
    # Get the heuristical maximum spatial distances among events with
    # respect to the timesteps.
    heuristical_steps = np.linspace(.1, .12, n_timesteps, dtype=np.float32, endpoint=False)[::-1]

    for t, s in enumerate(heuristical_steps):
        for i, e in enumerate(input_events_with_correlation_score):
            input_timestep = e[0][0]
            if input_timestep != t:
                continue
            is_out_of_reach = True
            for target_node in np.unique([ e[1] for e in target_events ]):
                input_node = e[0][1]
                if adj_matrix[input_node, target_node] <= s or adj_matrix[target_node, input_node] <= s:
                    is_out_of_reach = False
                    break
            if is_out_of_reach:
                del input_events_with_correlation_score[i]

    monte_carlo_tree_search = MonteCarloTreeSearch(
        spatial_temporal_gnn,
        navigator,
        scaler,
        torch.FloatTensor(x).to(device=spatial_temporal_gnn.device),
        torch.FloatTensor(y).to(device=spatial_temporal_gnn.device),
        maximum_leaf_size=maximum_leaf_size,
        exploration_weight=exploration_weight)

    root = Node(
        input_events=[ e for e, _ in input_events_with_correlation_score ])

    for i in range(n_rollouts):
        if verbose:
            print(f'Execution {i+1}/{n_rollouts}')
        monte_carlo_tree_search.rollout(root)
        if verbose:
            print('mae:', - monte_carlo_tree_search.best_leaf[1])
    
    return monte_carlo_tree_search.best_leaf[0]

def get_explanations_from_data(
    x: np.ndarray,
    y: np.ndarray,
    adj_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    navigator: Navigator,
    scaler: Scaler,
    n_rollouts: int = 20,
    explanation_size: Optional[int] = None,
    exploration_weight: int = 500,
    top_n_input_events: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:

    explained_x, explained_y = [], []

    for x_, y_ in zip(x, y):
        if explanation_size is None:
            explanation_size = min((y_.flatten() != 0).sum() * 2, 500)

        subset = get_best_input_subset(
            x_,
            y_,
            adj_matrix,
            spatial_temporal_gnn,
            navigator,
            scaler,
            n_rollouts=n_rollouts,
            maximum_leaf_size=explanation_size,
            exploration_weight=exploration_weight,
            top_n_input_events=top_n_input_events)

        # TODO: Fix this to add other events.
        input_events_subset = [ ( 0, e[0], e[1] ) for e in subset.input_events ]

        x_ = remove_features_by_events(x_, input_events_subset)

        x_ = x_[..., :1]

        explained_x.append(x_)
        explained_y.append(y_)

    return np.array(explained_x), np.array(explained_y)
