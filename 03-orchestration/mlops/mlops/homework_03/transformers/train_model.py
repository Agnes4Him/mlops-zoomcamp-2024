import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

@transformer
def train(data):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()
    train_dicts = data[categorical + numerical].to_dict(orient='records')

    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = data[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mean_squared_error(y_train, y_pred, squared=False)

    result = dict()
    result["lr"] = lr
    result["dv"] = dv

    return result
    