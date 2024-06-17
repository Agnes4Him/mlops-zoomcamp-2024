#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import os
import sys

print(f'loading model...')
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(url):
    df = pd.read_parquet(url)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def return_dataset(year, month, color):
    #url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month}.parquet'
    df = read_data(url)
    return df

def get_prediction(year, month, color):
    print(f'reading dataset {color}_tripdata for {month} {year}...')
    df = return_dataset(year, month, color)

    dicts = df[categorical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict(X)

    #print(np. std(y_pred))
    print(f'getting prediction {y_pred}')
    print(np.mean(y_pred))

    return y_pred

def get_rideid(year, month, color):
    df = return_dataset(year, month, color)
    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
    #df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    #print(df[:10])
    return df['ride_id']

def run(year, month, color):
    df_result = pd.DataFrame()

    df_result['ride_id'] = get_rideid(year, month, color)
    df_result['duration'] = get_prediction(year, month, color)

    print(f'checking if output directory already exist...')
    parent_dir = os.getcwd()
    dir = f'{parent_dir}/outputs/{color}/{year}'
    isExist = os.path.exists(dir)
    if not isExist:
        print(f'creating output directory for {month} {year} {color}_tripdata dataset...')
        os.makedirs(dir)

    output = f'{dir}/{month}.parquet'
    print(f'saving {output} file...')
    df_result.to_parquet(
        output,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == "__main__":
    #year = sys.argv[1]
    #month = sys.argv[2]
    #color = sys.argv[3]S
    year = os.getenv("YEAR")
    month = os.getenv("MONTH")
    color = os.getenv("COLOR")

    run(year, month, color)


