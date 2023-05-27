from typing import List, Optional, Tuple, Union
import numpy as np
import torch

def get_largest_event_set(
    data: np.ndarray) -> List[Tuple[int, Optional[int], Optional[int]]]:
    """
    Get the largest event set from the input spatial-temporal graph.
    Each event is represented by a tuple containing the event type, the
    time step at which the event occurs, and the node at which the event
    occurs. The second last element of the tuple is None if the event
    type is related to the time of day or the day of week. The last
    element of the tuple is None if the event type is related to the day
    of week.

    Parameters
    ----------
    data : ndarray
        The input spatial-temporal graph from which the largest event set is
        obtained.

    Returns
    -------
    list of (int, int | None, int | None)
        The largest event set from the input spatial-temporal graph.
    """
    # Get the number of time steps and nodes.
    n_time_steps, n_nodes, n_features = data.shape[-3:]

    # Get the largest event set related to the speed.
    speed_events = [
        (0, time_step, node) 
        for time_step in range(n_time_steps) 
        for node in range(n_nodes) 
        if data[..., time_step, node, 0] > 0]
    
    if n_features == 1:
        return speed_events

    # Get the largest event set related to the time of day.
    time_of_day_events = [
        (1, time_step, None) for time_step in range(n_time_steps)]

    # Get the largest event set related to the day of week.
    day_of_week_events = [(2, None, None)]

    return speed_events + time_of_day_events + day_of_week_events

def remove_features_by_events(
    data: Union[np.ndarray, torch.FloatTensor],
    events: List[Tuple[int, Optional[int], Optional[int]]]
    ) -> Union[np.ndarray, torch.FloatTensor]:
    """
    Remove the features of the input spatial-temporal graph that are
    not related to the events in the input event set.

    Parameters
    ----------
    data : ndarray | FloatTensor
        The input spatial-temporal graph from which the features are
        removed.
    events : list of (int, int | None, int | None)
        The input event set.

    Returns
    -------
    ndarray | FloatTensor
        The input spatial-temporal graph with the features not related
        to the events in the input event set removed.
    """
    if isinstance(data, torch.Tensor):
        filtered_data = data.clone()
    else:
        filtered_data = data.copy()
    # Get the number of time steps, nodes, and features.
    n_time_steps, n_nodes, n_features = filtered_data.shape[-3:]
    
    # Get the events related to the speed, time of day, and day of week.
    speed_events = [tuple(event) for event in events if event[0] == 0]
    time_of_day_events = [tuple(event) for event in events if event[0] == 1]
    day_of_week_events = [tuple(event) for event in events if event[0] == 2]
    
    # Remove the day of week features if there are no day of week events
    # and if the graph contains more than the sole speed feature.
    if n_features > 1 and not len(day_of_week_events):
        filtered_data[..., -7:] = 0
    
    # Loop through the time steps and nodes.
    for time_step in range(n_time_steps):
        for node in range(n_nodes):
            # Remove the speed features if there are no speed events
            # related to it.
            if (0, time_step, node) not in speed_events:
                filtered_data[..., time_step, node, 0] = 0

        # Remove the time of day features if there are no time of day
        # events related to them and if the graph contains more than the
        # sole speed feature.
        if n_features > 1 and (1, time_step, None) not in time_of_day_events:
            filtered_data[..., time_step, :, 1] = -1
    
    return filtered_data

def remove_single_event_from_data(
    data: Union[np.ndarray, torch.FloatTensor],
    event: Union[np.ndarray, torch.FloatTensor]
    ) -> Union[np.ndarray, torch.FloatTensor]:
    """
    Remove features from the input spatial-temporal graph according to
    the given input event.

    Parameters
    ----------
    data : ndarray | FloatTensor
        The input spatial-temporal graph from which the features are
        removed.
    event : ndarray | FloatTensor
        The input event that determines which features are removed from
        the input spatial-temporal graph.

    Returns
    -------
    ndarray | FloatTensor
        The input spatial-temporal graph with the features removed
        according to the input event.
    """
    # Get the kind, time step, and node of the input event.
    event_kind = event[0]
    time_step = int(event[1].item())
    node = int(event[2].item())

    # Remove the features from the input spatial-temporal graph
    # according to the input event.
    if event_kind == 0:
        data[..., time_step, node, 0] = 0
    elif event_kind == 1:
        data[..., time_step, :, 1] = -1
    elif event_kind == 2:
        data[..., -7:] = 0

    return data
