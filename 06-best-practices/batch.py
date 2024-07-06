#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd

#S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    '''folder_path = os.getcwd()
    output = "output"
    folder_path = os.path.join(folder_path, output) 

    print(f'creating {output} folder')
    
    os.makedirs(folder_path, exist_ok=True)''' 
    'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    #local_output_file = 'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def read_data(filename, categorical):
    if S3_ENDPOINT_URL == None:
        df = pd.read_parquet(filename)  
    else:
        df = pd.read_parquet(filename, storage_options=options)  
    df = prepare_data(df, categorical)

    return df

def save_data(df, output_file):
    if S3_ENDPOINT_URL == None:
        df.to_parquet(output_file, engine='pyarrow', index=False)
    else:
        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    print(f'loading model')

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    print(f'reading data from {input_file}')

    df = read_data(input_file, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('mean of predicted duration:', y_pred.mean())

    print('sum of predicted duration:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    print(f'writing predictions to {output_file}')

    save_data(df_result, output_file)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
