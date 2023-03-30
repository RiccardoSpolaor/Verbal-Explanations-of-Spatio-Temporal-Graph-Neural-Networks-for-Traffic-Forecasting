from math import ceil
from typing import List, Literal
from keplergl.keplergl import KeplerGl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_period = ['day_of_week', 'hour_of_day', 'week_of_year', 'day_of_year']

def group_dataframe_by_period(
    df: pd.DataFrame,
    period: Literal['day_of_week', 'hour_of_day', 'week_of_year', 'day_of_year'],
    aggregation: Literal['sum', 'mean']) -> pd.DataFrame:
    grouping_criteria = {
        'day_of_week': df.index.to_period('D'),
        'hour_of_day': df.index.to_period('H'),
        'week_of_year': df.index.to_period('W'),
        'day_of_year': df.index.to_period('D')
    }
    assert period in _period
    assert aggregation in ['sum', 'mean']
    grouped = df.groupby(grouping_criteria[period])
    if aggregation == 'sum':
        grouped = grouped.sum()
    elif aggregation == 'mean':
        grouped = grouped.mean()

    if period == 'day_of_week':
        grouped = grouped.groupby(grouped.index.day_of_week)
        grouped = grouped.mean()
    elif period == 'hour_of_day':
        grouped = grouped.groupby(grouped.index.hour)
        grouped = grouped.mean()
    return grouped

def get_node_values_with_location_dataframe(
    node_values_df: pd.DataFrame, locations_df: pd.DataFrame, metric_name: str,
    turn_datetimes_to_timestamp: bool) -> pd.DataFrame:
    node_values_location_df = pd.DataFrame(
    columns=['sensor_id', 'latitude', 'longitude', metric_name, 'datetime'])
    for c in node_values_df.columns.values:
        datetimes = node_values_df[c].index
        if turn_datetimes_to_timestamp:
            datetimes = datetimes.to_timestamp()

        missing_count = node_values_df[c].values

        latitude = locations_df.loc[locations_df['sensor_id'] == c].latitude.values[0]
        longitude = locations_df.loc[locations_df['sensor_id'] == c].longitude.values[0]
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
    node_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
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

    Returns
    -------
    DataFrame
        The resulting dataframe, useful for statistical evaluations.
    """
    # Get the number of timesteps, nodes and features.
    n_timesteps, n_nodes, n_features = node_values.shape

    assert len(feature_names) == n_features, \
    'The length of the `feature_names` list must correspond to the' +\
    'number of features in the `node_values` dataset.'

    # Reshape the `node_values` array as (T x N, F).
    node_values = np.reshape(node_values, (n_timesteps * n_nodes, n_features))

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
