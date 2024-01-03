import gc
from collections import defaultdict
from typing import List, Optional, Tuple
import multiprocess
import torch
import numpy as np

from ..events import remove_features_by_events
from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...data.data_processing import Scaler
from ..events import get_largest_event_set
from ..navigator.model import Navigator
from .monte_carlo_tree_copy import MonteCarloTreeSearch, Node, rollout


def get_best_input_subset(
    x: torch.FloatTensor,
    y: torch.FloatTensor,
    adj_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    navigator: Navigator,
    distance_matrix: np.ndarray,
    scaler: Scaler,
    n_rollouts: int = 50,
    maximum_leaf_size: int = 50,
    exploration_weight: float = 1_000,
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
        '''
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
        # Add the input event along with the correlation score to the list.'''
        input_events_with_correlation_score.append((e_, 0.))#correlation_score_avg))
        # Min max scale the correlation score.
        
        heuristical_score = []
        for t_e in target_events:
            heuristical_score.append(_get_CFL_constant(x, y, e_[0], t_e[0], e_[1], t_e[1], distance_matrix, adj_matrix))
            
        heuristical_score = np.mean(heuristical_score).item()
        input_events_with_correlation_score[-1] = (input_events_with_correlation_score[-1][0], heuristical_score)

    # Sort the input events with respect to the correlation score.
    input_events_with_correlation_score = sorted(
        input_events_with_correlation_score, key=lambda ev: ev[1])
    # Get the `top_n_input_events` input events.
    input_events_with_correlation_score = input_events_with_correlation_score[-top_n_input_events:]

    '''# Get the number of timesteps.
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
                del input_events_with_correlation_score[i]'''

    root = Node(
        input_events=[ e for e, _ in input_events_with_correlation_score ])
    
    from copy import deepcopy
    
    C = defaultdict(int)
    N = defaultdict(int)
    children = dict()
    expanded_children = dict()
    best_leaf = ( None, - math.inf )

    m = MonteCarloTreeSearch(
            maximum_leaf_size,
            exploration_weight,
            C,
            N,
            children,
            expanded_children,
            best_leaf)

    for i in range(n_rollouts):

        if verbose:
            print(f'Execution {i+1}/{n_rollouts}')

        with multiprocess.Pool(multiprocess.cpu_count()) as p:
            paths = p.starmap(rollout, [ (m, root) for _ in range(1) ])
        
        # print([p[-1] for p in paths])
        
        idx, reward = MonteCarloTreeSearch.reward([p[-1] for p in paths], spatial_temporal_gnn, x, y, scaler)
        
        m.backpropagate(paths[idx], reward)

        C = m.C
        N = m.N
        children = m.children
        expanded_children = m.expanded_children
        best_leaf = m.best_leaf

        if verbose:
            print('mae:', - reward)
        gc.collect()
        
    # best_leaf = monte_carlo_tree_search.best_leaf#[0]
    print(best_leaf[1])
    del monte_carlo_tree_searches
    del root
    gc.collect()
    
    return best_leaf
    '''monte_carlo_tree_search = MCTS(
        [ e for e, _ in input_events_with_correlation_score ],
        maximum_leaf_size,
        torch.FloatTensor(x).to(device=spatial_temporal_gnn.device),
        torch.FloatTensor(y).to(device=spatial_temporal_gnn.device),
        spatial_temporal_gnn,
        scaler)
    monte_carlo_tree_search.self_play(30)'''

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
        subset = get_best_input_subset(
            x_,
            y_,
            adj_matrix,
            spatial_temporal_gnn,
            navigator,
            distance_matrix,
            scaler,
            n_rollouts=n_rollouts,
            maximum_leaf_size=explanation_size_,
            exploration_weight=exploration_weight,
            top_n_input_events=top_n_input_events_)

        # TODO: Fix this to add other events.
        input_events_subset = [ ( 0, e[0], e[1] ) for e in subset.input_events ]

        x_ = remove_features_by_events(x_, input_events_subset)

        x_ = x_[..., :1]

        explained_x.append(x_)
        explained_y.append(y_)

    return np.array(explained_x), np.array(explained_y)

import math
# MPH_TO_KMH_FACTOR = 1.609344
CONGESTION_SPEED = 60 # mph #/ MPH_TO_KMH_FACTOR

def _get_CFL_constant(instance_x, instance_y, t_source, t_target, n_source, n_target, distance_matrix, adj_matrix):
    t, _, _ = instance_x.shape
    # Get the speed at the source node.
    speed_source = instance_x[t_source, n_source, 0]
    speed_target = instance_y[t_target, n_target, 0]
    
    if speed_target == 0:
        return 0

    # Given the time interval ids, get the time interval values in hours, knowing that
    # the time interval is 5 minutes.
    delta_time = (t_target + t - t_source) * 5 / 60

    # Do log base 50 in numpy
    #x = np.log(speed_target) / np.log(CONGESTION_SPEED)

    # Get the distance between the source and target nodes.
    #if adj_matrix[n_source, n_target] == 0:
    #    delta_distance = 100
    #else:
    delta_distance = distance_matrix[n_source, n_target]
    #print(type(speed_target.item()), speed_target.item())
    '''if speed_target.item() < CONGESTION_SPEED:
        k = 1 / math.log(speed_target.item(), CONGESTION_SPEED) * (1e-3)
    else:
        k = math.log(speed_target.item(), CONGESTION_SPEED) * 100
    #return k * ( speed_source * delta_time / (delta_distance + 1e-8) )
    if delta_distance == 0:
        delta_distance = np.min(distance_matrix.nonzero())
    return k * speed_source * delta_time / (delta_distance * 10)'''
    # Compute CFL constant 
    # - result lower than 1 means no congestion expected at target node
    # - result higher than 1 means congestion expected at target node
    C = speed_source * delta_time / (delta_distance + 1e-8)
    # If I don't have a target congestion
    # - I want to deprioritize nodes that are expected to lead to congestion (C > 1)
    # - I want to prioritize nodes that are expected to lead to no congestion (C < 1)
    # - I want to prioritize lower speeds, since it is easier to create a free flow of traffic if the traffic is flowing slower.
    if speed_target.item() >= CONGESTION_SPEED:
        return (1 / C) * (CONGESTION_SPEED / speed_target.item() )#math.log(speed_target.item(), CONGESTION_SPEED)
    # If I have a target congestion
    # - I want to deprioritize nodes that are expected to lead to no congestion (C < 1)
    # - I want to prioritize nodes that are expected to lead to congestion (C > 1)
    # - I want to prioritize higher speeds, since it is easier to get to congestion if the traffic is flowing faster.
    else:
        return C * (speed_target.item() / CONGESTION_SPEED)#math.log(speed_target.item(), CONGESTION_SPEED)

    #return (math.log(speed_target.item(), CONGESTION_SPEED) - 1) * (speed_source * delta_time / (delta_distance + 1e-1) - 1)

    #return (math.log(speed_target, CONGESTION_SPEED) - 1) * (speed_source * delta_time / (delta_distance + 1e-1) - 1)