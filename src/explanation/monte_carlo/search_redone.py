import math
from typing import List, Optional, Tuple
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np

from ..events import remove_features_by_events
from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...spatial_temporal_gnn.metrics import MAE
from ...data.data_processing import Scaler
from ...explanation.events import get_largest_event_set
from ...explanation.navigator.model import Navigator
from .monte_carlo_tree_redone import MonteCarloTreeSearch, Node


MPH_TO_KMH_FACTOR = 1.609344
CONGESTION_SPEED = 60 / MPH_TO_KMH_FACTOR

def get_best_input_subset(
    x: torch.FloatTensor,
    y: torch.FloatTensor,
    adj_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    distance_matrix: np.ndarray,
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
        
        scores = []
        
        for t_e in target_events:
            scores.append(_get_CFL_constant(x, y, e_[0], t_e[0], e_[1], t_e[1], distance_matrix))
        
        correlation_score_avg = np.mean(scores).item()

        # Add the input event along with the correlation score to the list.
        input_events_with_correlation_score.append((e_, correlation_score_avg))
        
    ### Global scoring.
    xs_with_removed_events = []
    
    ae_criterion = MAE(apply_average=False)
    
    for e in input_events_with_correlation_score:
        
        timestep, node = e[0][:2]
        # Remove the speed event at timestep i and node j.
        x_with_removed_event = torch.tensor(x).float().to(device=spatial_temporal_gnn.device)
        x_with_removed_event[timestep, node, 0] = 0.

        xs_with_removed_events.append(x_with_removed_event)
    
    with torch.no_grad():
        simulated_instances_loader = DataLoader(
            xs_with_removed_events,
            batch_size=64,
            shuffle=False)

        simulated_instances_scores = []
        for simulated_batch in simulated_instances_loader:
            simulated_batch = scaler.scale(simulated_batch)
            simulated_batch = simulated_batch.float().to(device=spatial_temporal_gnn.device)
            y_pred = spatial_temporal_gnn(simulated_batch)
            y_pred = scaler.un_scale(y_pred)

            # Repeat the target graph y by adding a batch dimension for the y_pred batch size.
            y_repeated = torch.tensor(y, dtype=torch.float32, device=spatial_temporal_gnn.device)
            y_repeated = y_repeated.unsqueeze(0).repeat(y_pred.shape[0], 1, 1, 1)

            simulated_instances_scores.append(ae_criterion(y_pred, y_repeated).cpu().numpy())
        simulated_instances_scores = np.concatenate(simulated_instances_scores)
        
        max_ae = simulated_instances_scores.max()
        min_ae = simulated_instances_scores.min()
        
        simulated_instances_scores = (max_ae - simulated_instances_scores) / (max_ae - min_ae)
        
        for i, (e, s) in enumerate(zip(input_events_with_correlation_score, simulated_instances_scores)):
            timestep, node = e[0][:2]
            # Add to the scoring of input events with correlation
            # score.
            input_events_with_correlation_score[i] = (input_events_with_correlation_score[i][0], input_events_with_correlation_score[i][1] * + s)
            #input_events_with_correlation_score.append(((timestep, node, 0), s))

    ### Global scoring end.

    # Sort the input events with respect to the correlation score.
    input_events_with_correlation_score = sorted(
        input_events_with_correlation_score, key=lambda x: x[1])
    # Get the `top_n_input_events` input events.
    input_events_with_correlation_score = input_events_with_correlation_score[-top_n_input_events:]

    # Get the number of timesteps.
    #n_timesteps = x.shape[-3]
    # Get the heuristical maximum spatial distances among events with
    # respect to the timesteps.
    '''heuristical_steps = np.linspace(.1, .12, n_timesteps, dtype=np.float32, endpoint=False)[::-1]

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
                del input_events_with_correlation_score[i]''';

    monte_carlo_tree_search = MonteCarloTreeSearch(
        spatial_temporal_gnn,
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

def _get_CFL_constant(instance_x, instance_y, t_source, t_target, n_source, n_target, distance_matrix):
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
    delta_distance = distance_matrix[n_source, n_target]
    #print(type(speed_target.item()), speed_target.item())
    return (speed_target.item() - CONGESTION_SPEED) * (speed_source * delta_time / (delta_distance + 1e-1) - 1)
    #return (math.log(speed_target, CONGESTION_SPEED) - 1) * (speed_source * delta_time / (delta_distance + 1e-1) - 1)