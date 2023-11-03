from typing import List, Optional, Tuple, Union
import numpy as np
import torch

def get_largest_event_set(
    x: np.ndarray
    ) -> np.ndarray:
    """
    Get the largest event set from the input spatial-temporal graph.
    An event is defined as a positive value in the speed feature of the
    input spatial-temporal graph.
    Each event is represented by a tuple containing, the
    time step at which the event occurs, and the node at which the event
    occurs. 

    Parameters
    ----------
    x : ndarray
        The input spatial-temporal graph from which the largest event set is
        obtained.

    Returns
    -------
    ndarray
        The largest event set from the input spatial-temporal graph.
    """
    return [tuple(e) for e in np.argwhere(x[..., 0] > 0)]

def remove_features_by_events(
    x: Union[np.ndarray, torch.FloatTensor],
    events: np.ndarray,
    remove_value = 0.
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
    if isinstance(x, torch.Tensor):
        x = x.clone()
        # Create a tensor of the shape of x with all remove_value
        filtered_x = torch.full_like(x[..., 0], remove_value)
    else:
        x = x.copy()
        filtered_x = np.full_like(x[..., 0], remove_value)

    filtered_x[x[..., 0] == 0] = remove_value

    for e in events:
        filtered_x[e[0], e[1]] = x[e[0], e[1], 0]
    
    x[..., 0] = filtered_x
    return x
    
    
        
    
    
    events_to_keep = [tuple([event[0], event[1], 0]) for event in events] + [tuple(i) for i in positive_indices]
    
    # On indices not in events to keep, put remove_value
    if isinstance(x, torch.Tensor):
        print(~torch.tensor(events_to_keep))
        filtered_x[~torch.tensor(events_to_keep)] = remove_value
    else:
        filtered_x[~np.array(events_to_keep)] = remove_value
    return filtered_x
        
        
    # On indices not in events, put remove_value if the feature is not zero
    filtered_x[filtered_x[..., 0] != 0] = remove_value
    
        
    '''# Put remove value on the speed features that are not related to
    # the events in the input event set.
    speed_events = events[events[:, 0] == 0]
        
    np.argwhere(filtered_x[..., 0] > 0)
    # Get the number of time steps, nodes, and features.
    n_time_steps, n_nodes, _ = filtered_data.shape[-3:]
    
    # Get the events related to the speed, time of day, and day of week.
    #speed_events = [tuple(event) for event in events if event[0] == 0]
    #time_of_day_events = [tuple(event) for event in events if event[0] == 1]
    #day_of_week_events = [tuple(event) for event in events if event[0] == 2]
    
    # Remove the day of week features if there are no day of week events
    # and if the graph contains more than the sole speed feature.'''
    '''if n_features > 1 and not len(day_of_week_events):
        filtered_data[..., -7:] = 0'''
    
    '''# Loop through the time steps and nodes.
    for time_step in range(n_time_steps):
        for node in range(n_nodes):
            # Remove the speed features if there are no speed events
            # related to it.
            if (0, time_step, node) not in speed_events:
                filtered_data[..., time_step, node, 0] = remove_value'''

    ''' # Remove the time of day features if there are no time of day
        # events related to them and if the graph contains more than the
        # sole speed feature.
        if n_features > 1 and (1, time_step, None) not in time_of_day_events:
            filtered_data[..., time_step, :, 1] = -1'''
    
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
