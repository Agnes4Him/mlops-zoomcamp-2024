from datetime import datetime
import pandas as pd
from batch import *


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def get_actual_df():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

    categorical = ['PULocationID', 'DOLocationID']

    df = pd.DataFrame(data, columns=columns)

    actual_df = prepare_data(df, categorical)

    return actual_df


def get_expected_df():
    data = [
        (-1, -1, dt(1, 1), dt(1, 10), 9.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']

    expected_df = pd.DataFrame(data, columns=columns)

    return expected_df


def test_duration():
    actual_df = get_actual_df()

    expected_df = get_expected_df()

    assert actual_df['duration'].all() == expected_df['duration'].all()

def test_pickup_location():
    actual_df = get_actual_df()

    expected_df = get_expected_df()
    
    assert actual_df['PULocationID'].all() == expected_df['PULocationID'].all()

def test_dropoff_location():
    actual_df = get_actual_df()

    expected_df = get_expected_df()
    
    assert actual_df['DOLocationID'].all() == expected_df['DOLocationID'].all()

def test_pickup_datetime():
    actual_df = get_actual_df()

    expected_df = get_expected_df()
    
    assert actual_df['tpep_pickup_datetime'].all() == expected_df['tpep_pickup_datetime'].all()

def test_dropoff_datetime():
    actual_df = get_actual_df()

    expected_df = get_expected_df()
    
    assert actual_df['tpep_dropoff_datetime'].all() == expected_df['tpep_dropoff_datetime'].all()

