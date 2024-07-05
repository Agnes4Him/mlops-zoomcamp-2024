#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df = prepare_data(df, categorical)

    return df


def main(year, month):
    '''isExist = os.path.exists(f'{os.getcwd()}/output')
    if not isExist:
        os.makedirs(f'{os.getcwd()}/output')'''
    folder_path = os.getcwd()
    output = "output"
    folder_path = os.path.join(folder_path, output) 

    print(f'creating {output} folder')
    
    os.makedirs(folder_path, exist_ok=True) 

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'{output}/yellow_tripdata_{year:04d}-{month:02d}.parquet'

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


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    print(f'writing predictions to {output_file}')

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)