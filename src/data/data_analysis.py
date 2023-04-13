import json
from math import ceil
from typing import get_args, List, Literal, Optional
from keplergl.keplergl import KeplerGl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd


def get_node_values_with_location_dataframe(
    node_values_df: pd.DataFrame, locations_df: pd.DataFrame, metric_name: str,
    turn_datetimes_to_timestamp: bool) -> pd.DataFrame:
    """Get a pandas dataframe from a pandas dataframe of node speed values
    and a pandas dataframe of node locations. The resulting dataframe
    has for each timestamp the value of the metric for each node and
    the location of the node in the form of latitude and longitude.

    Parameters
    ----------
    node_values_df : DataFrame
        The dataframe containing the values of speed for each node.
    locations_df : DataFrame
        The dataframe containing the location of each node.
    metric_name : str
        The name of the metric that will be used in the resulting dataframe.
    turn_datetimes_to_timestamp : bool
        Whether to turn the datetimes to timestamp or not.

    Returns
    -------
    DataFrame
        The resulting dataframe containing the values of the metric for
        each node and the location of the node in the form of latitude
        and longitude.
    """
    node_values_location_df = pd.DataFrame(
    columns=['sensor_id', 'latitude', 'longitude', metric_name, 'datetime'])
    
    for c in node_values_df.columns.values:
        datetimes = node_values_df[c].index
        if turn_datetimes_to_timestamp:
            datetimes = datetimes.to_timestamp()

        missing_count = node_values_df[c].values

        latitude = locations_df.loc[
            locations_df['sensor_id'] == c].latitude.values[0]
        longitude = locations_df.loc[
            locations_df['sensor_id'] == c].longitude.values[0]
        latitudes = [latitude] * len(missing_count)
        longitudes = [longitude] * len(missing_count)

        node_timeseries_df = pd.DataFrame({
            'sensor_id': [c] * len(missing_count),
            'latitude': latitudes,
            'longitude': longitudes,
            metric_name: missing_count,
            'datetime': datetimes
        })
        node_values_location_df = pd.concat(
            [node_values_location_df, node_timeseries_df],
            axis=0, ignore_index=True)

    return node_values_location_df

def get_node_values_statistics_dataframe(
    node_values: np.ndarray, feature_names: List[str],
    has_day_of_the_week: bool) -> pd.DataFrame:
    """
    Get a pandas dataframe from a numpy array of features
    which is useful for statistical evaluations.
    The resulting dataframe will contain for each row an instance
    of a node at a certain timestep and the columns will present
    the node features.

    Parameters
    ----------
    node_values : ndarray
        The dataset containing the speed values of the nodes at different
        timestamps. Its shape is (T, N, F), where:
        * T is the number of timesteps.
        * N is the number of nodes.
        * F is the number of features.
    feature_names : list of str
        The name of the features that will be displayed in the resulting
        statistics dataframe.
    has_day_of_the_week : bool
        Whether the dataset has a day of the week one-hot encoded feature
        vector or not.

    Returns
    -------
    DataFrame
        The resulting dataframe, useful for statistical evaluations.
    """
    # Get the number of timesteps, nodes and features.
    n_timesteps, n_nodes, n_features = node_values.shape

    # Reshape the `node_values` array as (T x N, F).
    node_values = np.reshape(node_values, (n_timesteps * n_nodes, n_features))

    if has_day_of_the_week:
        # Transform the day of the week one-hot encoded feature vector
        # to a single integer representing the day of the week.
        day_of_the_week = np.argmax(node_values[:, -7:], axis=1)
        node_values = node_values[:, :-7]
        node_values = np.concatenate(
            (node_values, day_of_the_week[:, None]), axis=1)

    # Get the statistics dataframe.
    return pd.DataFrame(node_values, columns=feature_names)

def plot_features_distribution(statistics_df: pd.DataFrame, bins: List[int],
                               x_labels: List[str], title: str) -> None:
    """
    Plot the distribution of each column feature from a dataframe
    collecting the feature information of each observation.

    Parameters
    ----------
    statistics_df : DataFrame
        The input dataframe.
    bins : list of int
        The bin size to use to discretize each feature values.
    x_labels : list of str
        The label to put in the x axis of the resulting graph of each
        feature distribution.
    title : str
        The title of the plot.
    """
    assert len(statistics_df.columns) == len(bins) == len(x_labels), \
    'The number of `bins` and `x_labels` must match the number of ' +\
    'dataframe columns.'

    # Set the plot rows and columns number according to the number 
    # of features in the dataframe. 
    n_features = len(statistics_df.columns)
    n_cols = 2
    n_rows = ceil(n_features / 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    # Set the axes indices.
    axes_indices = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    # Plot the subplots of the distribution of each feature.
    for c, ax, b, l in zip(statistics_df.columns, axes_indices, bins, x_labels):
        statistics_df[c].plot(ax=axes[ax], kind='hist', edgecolor='black',
                              bins=b, title=f'Distribution of feature "{c}"')
        axes[ax].set_xlabel(l)

    # Delete eventual extra subplots.
    for ax in axes_indices[len(statistics_df.columns):]:
        fig.delaxes(axes[ax])

    # Set the main title.
    plt.suptitle(title)
    plt.show()

def get_missing_values_by_location_dataframe(
    node_values_df: pd.DataFrame, locations_df: pd.DataFrame) -> pd.DataFrame:
    """Get a dataframe containing the missing values ratio per node.

    Parameters
    ----------
    node_values_df : DataFrame
        The input dataframe containing the speed values of the nodes
        at different timestamps.
    locations_df : DataFrame
        The input dataframe containing the locations of each node in
        terms of latitude and longitude.

    Returns
    -------
    DataFrame
        The resulting dataframe containing the missing values ratio of
        each node along with its location.
    """
    missing_values_per_node_ratio = {
        c: (node_values_df[c] == 0.).mean(axis=0)
        for c in node_values_df.columns}
    
    locations_df = locations_df.copy()

    locations_df['Total missing values'] = locations_df.apply(
        lambda x: missing_values_per_node_ratio[x[0]], axis = 1
    )
    
    return locations_df

def show_kepler_map(
    data: pd.DataFrame, config_file_path: Optional[str] = None,
    height: int = 800) -> KeplerGl:
    """Get a Kepler map based on the input data and its configuration.

    Parameters
    ----------
    data : DataFrame
        The input dataframe containing the data to be visualized.
    config_file_path : str, optional
        The configuration file path to style the map, by default None.
    height : int, optional
        The height of the window that contains the map, by
        default 800.

    Returns
    -------
    KeplerGl
        The resulting Kepler map.
    """
    if config_file_path is not None:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    return KeplerGl(height=height, config=config, show_docs=False,
                    data={'data': data})

DayOfTheWeek = Literal['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                       'Saturday', 'Sunday']

days_encoder = { d: i for i, d in enumerate(get_args(DayOfTheWeek)) }
days_decoder = { i: d for i, d in enumerate(get_args(DayOfTheWeek)) }


def get_day_dataframe(
    node_values_df: pd.DataFrame, day: DayOfTheWeek,
    remove_off_peak_hours: bool = True) -> pd.DataFrame:
    """Get a dataframe containing the speed vlue of the nodes for a
    specific day of the week first occurence, possibly removing the
    off-peak hours.

    Parameters
    ----------
    node_values_df : DataFrame
        The input dataframe containing the speed values of the nodes.
    day : 'Monday' | 'Tuesday' | 'Wednesday' | 'Thursday' | 'Friday'
    | 'Saturday' | 'Sunday'
        The day to consider.
    remove_off_peak_hours : bool, optional
        Whether to remove the off-peak hours, by default True.

    Returns
    -------
    DataFrame
        The resulting dataframe containing the speed values of the
        first occurence of the specified day of the week, with the
        peak hours removed if specified.
    """
    assert day in days_encoder.keys(), f'Invalid day: {day}.'

    day_id = days_encoder[day]
    days_df = node_values_df[node_values_df.index.dayofweek == day_id]

    first_day_date = days_df.index.to_period('D')[0]
    first_day_df = days_df[days_df.index.to_period('D') == first_day_date]

    if remove_off_peak_hours:
        first_day_df = first_day_df[first_day_df.index.hour.isin(range(7, 20))]

    return first_day_df

def get_node_values_df_of_nodes_with_largest_speed_variation(
    node_values_df: pd.DataFrame, n_nodes: int = 3) -> pd.DataFrame:
    """Get the dataframe containing the speed values of the nodes with
    largest speed variation.

    Parameters
    ----------
    node_values_df : DataFrame
        The input dataframe containing the speed values of the nodes.
    n_nodes : int
        The number of nodes to consider, by default 3.

    Returns
    -------
    DataFrame
        The resulting dataframe containing the speed values of the
        nodes with largest speed variation.
    """
    node_values_std_df = node_values_df.replace(0, np.NaN).std(skipna=True)
    nodes_with_largest_variation = \
        node_values_std_df.sort_values()[-n_nodes:].index.to_list()
    
    return node_values_df[nodes_with_largest_variation]

def plot_daily_speed_variation(node_values_reduced: pd.DataFrame) -> None:
    """Plot the average speed of nodes for each day of the week
    aggregated by hour.

    Parameters
    ----------
    node_values_reduced : DataFrame
        The input dataframe containing the speed values of the nodes
        with the largest speed variation over the period.
    """
    # Create a figure with multiple subplots.
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    # Define the indices of the subplots.
    subplot_indices = [(r, c) for r in range(3) for c in range(3)]

    # Plot the average speed of nodes for each day of the week and each node.
    for day_index, subplot_index in zip(range(7), subplot_indices):
        # Filter the data for the current day of the week.
        day_data = node_values_reduced[node_values_reduced.index.dayofweek == day_index]

        # Get the first occurrence of the selected day and filter the data.
        first_day = day_data.index.to_period('D')[0]
        first_day_data = day_data[day_data.index.to_period('D') == first_day]
        first_day_data = first_day_data.groupby(first_day_data.index.hour).mean()

        # Plot the mean speed for each node.
        for n in first_day_data.columns:
            node_df = first_day_data[n]
            if day_index != 0:
                label=None
            else:
                label=f'Node ID: {n}'
            axes[subplot_index].plot(node_df.index.values, node_df.values, label=label)

        # Set the axis labels and tick marks.
        axes[subplot_index].set_title(days_decoder[day_index])
        axes[subplot_index].set_xlabel('Hour')
        axes[subplot_index].set_ylabel('Speed (km/h)')
        ticks_loc = axes[subplot_index].get_xticks().tolist()
        xticks = [f'{int(t):02d}:00' for t in ticks_loc]
        axes[subplot_index].xaxis.set_major_locator(FixedLocator(ticks_loc))
        axes[subplot_index].set_xticklabels(xticks)

    # Remove any extra subplots.
    for subplot_index in subplot_indices[7:]:
        fig.delaxes(axes[subplot_index])

    # Add a title and legend to the figure.
    plt.suptitle('Speed of the nodes with the largest variation of the feature during the first week.', fontsize=14)
    plt.tight_layout()
    plt.figlegend(loc='lower center', ncol=1, bbox_to_anchor=(0.52, 0.15), fontsize=12)

    # Show the figure.
    plt.show()

def plot_average_speed_by_hour(node_values_df: pd.DataFrame) -> None:
    """Plot the average speed of nodes aggregated by hour.

    Parameters
    ----------
    node_values_df : DataFrame
        The input dataframe containing the speed values of the nodes.
    """
    # Replace the 0 values with NaNs.
    node_speed_with_nans = node_values_df.replace(0, np.NaN)
    # Group the data by hour of the day and compute the mean.
    grouped_by_hour_of_day = _get_grouped_dataframe_by_period(
        node_speed_with_nans, 'hour_of_day')
    grouped_mean = grouped_by_hour_of_day.mean(axis=1, skipna=True)
    
    plt.figure(figsize=(15, 10))
    plt.plot([f'{int(t):02d}:00' for t in grouped_mean.index],
             grouped_mean.values)
    plt.xticks(plt.xticks()[0][::2])
    plt.title('Average speed per hour')
    plt.show()
    
def plot_average_speed_by_day(node_values_df: pd.DataFrame) -> None:
    """Plot the average speed of nodes aggregated by day.

    Parameters
    ----------
    node_values_df : DataFrame
        The input dataframe containing the speed values of the nodes.
    """
    # Replace the 0 values with NaNs.
    node_speed_with_nans = node_values_df.replace(0, np.NaN)
    # Group the data by hour of the day and compute the mean.
    grouped_by_hour_of_day = _get_grouped_dataframe_by_period(
        node_speed_with_nans, 'day_of_week')
    grouped_mean = grouped_by_hour_of_day.mean(axis=1, skipna=True)
    
    plt.figure(figsize=(15, 10))
    plt.plot(grouped_mean.index.map(days_decoder), grouped_mean.values)
    plt.title('Average speed per week day')
    plt.show()
    
GroupingPeriod = Literal['day_of_week', 'hour_of_day']

def _get_grouped_dataframe_by_period(
    df: pd.DataFrame, period: GroupingPeriod) -> pd.DataFrame:
    """Group the dataframe by the specified period.

    Parameters
    ----------
    df : DataFrame
        The input dataframe to be grouped.
    period : 'day_of_week' | 'hour_of_day'
        The period by which the dataframe will be grouped.

    Returns
    -------
    DataFrame
        The grouped dataframe.
    """
    grouping_criteria = {
        'day_of_week': df.index.day_of_week,
        'hour_of_day': df.index.hour,
    }
    assert period in get_args(GroupingPeriod), \
    'The `period` parameter must be one of the following: ' +\
    ', '.join(get_args(GroupingPeriod)) + '.'

    # Group the dataframe by the specified period.
    grouped = df.groupby(grouping_criteria[period])
    # Get the mean of each group.
    grouped = grouped.mean()

    return grouped