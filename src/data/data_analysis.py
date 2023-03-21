from typing import Literal
from keplergl.keplergl import KeplerGl
import matplotlib.pyplot as plt
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
