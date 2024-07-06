from datetime import datetime
import pandas as pd
import os
from batch import *

S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', "http://localhost:4566")
 #S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def write_df():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

    df = pd.DataFrame(data, columns=columns)

    input_file = get_input_path(2023, 1)
    
    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    print('parquet file has been added to localstack S3 bucket')

def read_dataframe(year, month):
    print(f'fetching dataframe for {month} {year} from localstack...')
    output_file = get_output_path(2023, 1)

    df = pd.read_parquet(output_file, storage_options=options)

    print(df)

write_df()

read_dataframe(2023, 1)