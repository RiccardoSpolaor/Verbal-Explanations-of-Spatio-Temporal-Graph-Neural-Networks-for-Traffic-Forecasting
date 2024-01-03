from typing import List, Literal, Optional, Tuple, Union
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
    remove_value: Union[float, Literal['perturb']] = 0.
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
        if remove_value == 'perturb':
            filtered_x = torch.normal(mean=0, std=5, size=x[..., 0].shape, device=x.device) + x[..., 0]
        else:
            filtered_x = torch.full_like(x[..., 0], remove_value)
    else:
        x = x.copy()
        # Create an array of the shape of x with all remove_value
        if remove_value == 'perturb':
            filtered_x = np.random.normal(loc=0, scale=5, size=x[..., 0].shape) + x[..., 0]
        else:
            filtered_x = np.full_like(x[..., 0], remove_value)

    # Put as 0 all the original missing speed events.
    filtered_x[x[..., 0] == 0] = 0.

    # Re-introduce the speed features that are related to the events.
    for e in events:
        filtered_x[e[0], e[1]] = x[e[0], e[1], 0]

    # Put the filtered speed features back into the input graph.
    x[..., 0] = filtered_x
    return x

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
