from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

from ..utils.config import (
    CONGESTION_THRESHOLD_MPH,
    SEVERE_CONGESTION_THRESHOLD_MPH,
    MPH_TO_KMH_FACTOR)


def get_cluster_type(x: np.ndarray) -> str:
    """Get the type of the cluster.

    Parameters
    ----------
    x : ndarray
        The cluster data.

    Returns
    -------
    str
        The type of the cluster.
    """
    if x.mean() <= SEVERE_CONGESTION_THRESHOLD_MPH * MPH_TO_KMH_FACTOR:
        return 'severe congestion'

    if x.mean() <= CONGESTION_THRESHOLD_MPH * MPH_TO_KMH_FACTOR:
        return 'congestion'

    return 'free flow'

def get_repetition_of_cluster_type_information(
    cluster_type_information: str,
    previous_cluster_type_information: List[str],
    ) -> int:
    """
    Get the number of times the cluster type information has been repeated
    in the previous clusters.

    Parameters
    ----------
    cluster_type_information : str
        The cluster type information.
    previous_cluster_type_information : list of str
        The list of the previous cluster type information.

    Returns
    -------
    int
        The number of times the cluster type information has been repeated
        in the previous clusters.
    """
    equal_clusters_count = 0
    for c in previous_cluster_type_information:
        if cluster_type_information == c:
            equal_clusters_count += 1
    return equal_clusters_count

def get_cluster_time_info(
    time_info: np.ndarray,
    time_indices: np.ndarray
    ) -> Dict[str, str]:
    """
    Get the time information of the cluster.

    Parameters
    ----------
    time_info : ndarray
        The array containing the time information of the nodes.
    time_indices : ndarray
        The indices of the nodes involved in the cluster.

    Returns
    -------
    { str: str }
        The dictionary containing the time information of the cluster.
    """
    # Get the minimum and maximum timestep of the target nodes.
    min_timestep, max_timestep = np.min(time_indices), np.max(time_indices)
    y_min_time, y_max_time = time_info[min_timestep][0], time_info[max_timestep][0]
    beginning_day, beginning_date, beginning_hour = _get_time(y_min_time)
    end_day, end_date, end_hour = _get_time(y_max_time)

    cluster_time_info = {}

    # Put the date and day information of the target nodes in the
    # knowledge graph.
    if beginning_date == end_date:
        cluster_time_info['on date'] = beginning_date
        cluster_time_info['on day'] = beginning_day
        if beginning_hour == end_hour:
            cluster_time_info['on time'] = beginning_hour
        else:
            cluster_time_info['from time'] = beginning_hour
            cluster_time_info['to time'] = end_hour
    else:
        cluster_time_info['from date'] = beginning_date
        cluster_time_info['to date'] = end_date

        cluster_time_info['from day'] = beginning_day
        cluster_time_info['to day'] = end_day

        # Put the time information of the target nodes in the knowledge graph.
        cluster_time_info['from time'] = beginning_hour
        cluster_time_info['to time'] = end_hour

    return cluster_time_info

def _get_time(date: np.datetime64) -> Tuple[str, str, str]:
    """Get the formatted time information from a datetime.

    Parameters
    ----------
    date : datetime64
        The datetime.

    Returns
    -------
    str
        The day of the week extracted from the datetime.
    str
        The date in the format DD/MM/YYYY extracted from the datetime.
    str
        The time in the format HH:MM extracted from the datetime.
    """
    days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    Y, M, D, h, m = [date.astype('datetime64[%s]' % kind) for kind in 'YMDhm']

    year = Y.astype(int) + 1970
    month = M.astype(int) % 12 + 1
    day = (D - M).astype(int) + 1
    day_of_week = days[((D - M).astype(int) - 1) % 7]
    hour = (h - D).astype(int)
    minute = (m - h).astype(int)

    return day_of_week, f'{day:02d}/{month:02d}/{year}', f'{hour:02d}:{minute:02d}'

def get_cluster_location_info(
    node_info: Dict[str, Tuple[str, int]],
    node_indices: np.ndarray,
    node_pos_dict: Dict[int, str],
    ) -> Dict[str, List[int]]:
    """
    Get a dictionary containing the street involved in the cluster along
    with their involved kms.

    Parameters
    ----------
    node_info : { str: (str, int) }
        The dictionary containing for each node id the street and kilometrage.
    node_indices : ndarray
        The indices of the nodes involved in the cluster.

    Returns
    -------
    { str: list of int }
        The dictionary containing the streets involved in the cluster along
        with their involved kms.
    """
    # Get the unique node indices.
    node_indices = np.unique(node_indices)
    # Get the IDs of the nodes by their indices.
    node_ids = [ node_pos_dict[idx] for idx in node_indices ]

    # Get a dictionary containing the street and kilometrage of each node.
    streets = {}
    for node_id in node_ids:
        # Get the street and kilometrage of the node.
        street, km = node_info[node_id]
        # Add the street and kilometrage to the dictionary.
        if not street in streets:
            streets[street] = [km]
        else:
            streets[street].append(km)
    # Sort the kms of each street and round them to the nearest integer.
    for street, kms in streets.items():
        streets[street] = sorted(set([int(km) for km in kms]))
    # Sort the streets by the number of involved kms.
    streets = dict(sorted(streets.items(), key=lambda x: len(x[1]), reverse=True))
    return streets

def get_repetition_of_location_information(
    location_information: Dict[str, List[int]],
    previous_location_information: List[Dict[str, List[int]]] = None,
    ) -> Dict[str, int]:
    """
    For each street, count the times it has been involved in the previous
    location information.

    Parameters
    ----------
    location_information : { str: list of int }
        The dictionary containing the location information.
    previous_location_information : list of { str: list of int }, optional
        The list of the previous location information, by default None

    Returns
    -------
    { str: int }
        The dictionary containing the times each street has been involved in
        the previous location information.
    """
    # Set the dictionary containing the times each street has been involved in
    # the previous location information to count 0 for each street.
    equal_times_counts = { k: 0 for k in location_information.keys() }

    if previous_location_information is not None:
        # For each street, count the times it has been involved in the previous
        # location information.
        for k in location_information.keys():
            equal_times_count = 0
            for previous_location_info in previous_location_information:
                if k in previous_location_info.keys():
                    equal_times_count += 1
            equal_times_counts[k] = equal_times_count

    return equal_times_counts


def get_cluster_time_span(
    temporal_information: Dict[str, str],
    ) -> Tuple[datetime, datetime]:
    """
    Get the time span of the cluster.

    Parameters
    ----------
    temporal_information : { str: str}
        The dictionary containing the time information of the cluster.

    Returns
    -------
    datetime
        The time datetime of the beginning of the cluster.
    datetime
        The time datetime of the end of the cluster.
    """
    if 'from time' in temporal_information:
        from_time = temporal_information['from time']
        from_time_datetime = datetime.strptime(from_time, '%H:%M')
        to_time = temporal_information['to time']
        to_time_datetime = datetime.strptime(to_time, '%H:%M')
        return (from_time_datetime.time(), to_time_datetime.time())
    else:
        on_time = temporal_information['on time']
        time_datetime = datetime.strptime(on_time, '%H:%M')
        return (time_datetime.time(), time_datetime.time())

def get_cluster_day_span(
    temporal_information: Dict[str, str],
    ) -> Tuple[datetime, datetime]:
    """
    Get the day span of the cluster.

    Parameters
    ----------
    temporal_information : { str: str }
        The dictionary containing the time information of the cluster.

    Returns
    -------
    datetime
        The day datetime of the beginning of the cluster.
    datetime
        The day datetime of the end of the cluster.
    """
    # Get the clusters sorted by time.
    if 'from day' in temporal_information:
        from_date = temporal_information['from date']
        from_date_datetime = datetime.strptime(from_date, '%d/%m/%Y')
        to_date = temporal_information['to date']
        to_date_datetime = datetime.strptime(to_date, '%d/%m/%Y')
        from_time, to_time = get_cluster_time_span(temporal_information)
        return (datetime.combine(from_date_datetime, from_time),
                datetime.combine(to_date_datetime, to_time))
    else:
        on_date = temporal_information['on date']
        on_date_datetime = datetime.strptime(on_date, '%d/%m/%Y')
        from_time, to_time = get_cluster_time_span(temporal_information)
        return (datetime.combine(on_date_datetime, from_time),
                datetime.combine(on_date_datetime, to_time))
