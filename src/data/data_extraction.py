import pandas as pd


MPH_TO_KMH_FACTOR = 1.609344

def get_node_values_dataframe(
    hdf_file_path: str, turn_in_kmph: bool = True) -> pd.DataFrame:
    # Get the dataframe containing the node values at different timestamps.
    node_values_df = pd.read_hdf(hdf_file_path)

    if turn_in_kmph:
        # Transform the speed from miles/h to km/h.
        node_values_df.iloc[:] = node_values_df.iloc[:] * MPH_TO_KMH_FACTOR
        
    node_values_df.columns = [str(c) for c in node_values_df.columns]

    return node_values_df

def get_locations_dataframe(
    hdf_file_path: str, has_header: bool) -> pd.DataFrame:
    # Get the dataframe containing the latitude and longitude of each sensor.
    if has_header:
        locations_df = pd.read_csv(hdf_file_path, index_col='index')
    else:
        locations_df = pd.read_csv(
            hdf_file_path, names=['sensor_id', 'latitude', 'longitude'])
        locations_df.index.name = 'index'

    # Set the sensor_id feature as string values.
    locations_df['sensor_id'] = locations_df['sensor_id'].apply(str)
    return locations_df
