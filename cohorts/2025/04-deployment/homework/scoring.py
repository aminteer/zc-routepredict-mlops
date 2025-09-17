#!/usr/bin/env python


import pickle
import pandas as pd
import argparse

# parse arguments
p = argparse.ArgumentParser()
p.add_argument("--year", type=int, default=2022, help="year of the data")
p.add_argument("--month", type=int, default=1, help="month of the data")
args = p.parse_args()

# define categorical features
categorical = ['PULocationID', 'DOLocationID']

# function to load model
def load_model(model_path='model.bin'):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


dv, model = load_model('model.bin')

# function to read and preprocess data
def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def make_predictions(df, dv, model):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def save_results(df, y_pred, year, month, output_file):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['prediction'] = y_pred
    df_result = df[['ride_id', 'prediction']]
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    
def run(year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'results_{year:04d}-{month:02d}.parquet'
    
    df = read_data(input_file)
    y_pred = make_predictions(df, dv, model)
    save_results(df, y_pred, year, month, output_file)
    
    
if __name__ == "__main__":
    run(args.year, args.month)

    # For debugging purposes

    # print(y_pred)

    # st_dev = y_pred.std()
    # print(f'Standard Deviation of predictions: {st_dev}')



