from datetime import datetime
import pandas as pd
from batch import *

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']

    actual = prepare_data(df, categorical)

    test_data = [
        (-1, -1, dt(1, 1), dt(1, 10), 9),
        (1, 1, dt(1, 2), dt(1, 10), 8),
    ]

    test_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']

    expected = pd.DataFrame(test_data, columns=test_columns)

    assert actual == expected
