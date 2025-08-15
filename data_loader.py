import os
import pandas as pd
import numpy as np
from numpy import loadtxt

def load_dataset(dataset_name):

    file_path = os.path.join('data', f'{dataset_name}')

    if dataset_name == 'taiwan_bankrupt.csv':
        dataset = pd.read_csv(file_path, delimiter=",", skiprows=1)
        y = dataset.iloc[:, 0]
        X = dataset.iloc[:, 1:]
        X = X.values
        y = y.values

    elif dataset_name == 'diabetes.csv':
        dataset = loadtxt(file_path, delimiter=",", skiprows=1)
        X = dataset[0:, 1:21]
        y = dataset[0:, 0]
        pass

    elif dataset_name == 'diabetes1.csv':
        dataset = loadtxt(file_path, delimiter=",", skiprows=1)
        X = dataset[0:, 1:21]
        y = dataset[0:, 0]
        pass

    elif dataset_name == 'magic04.data':
        dataset = pd.read_csv(file_path, delimiter=",")
        X = dataset.iloc[:, 0:10]
        y = dataset.iloc[:, 10]
        y = y.replace({"g": 1, "h": 0})
        X = X.values
        y = y.values
        pass

    elif dataset_name == 'bank-additional.csv':
        dataset = pd.read_csv(file_path, delimiter=";")
        dataset["y"] = dataset["y"].replace({"yes": 1, "no": 0})
        X = dataset.drop(columns="y")
        y = dataset["y"]
        from sklearn.preprocessing import OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
        X = ordinal_encoder.fit_transform(X)
        pass

    elif dataset_name == 'HTRU_2.csv':
        dataset = pd.read_csv(file_path, sep=',', skiprows=0)
        X = dataset.iloc[:, 0:8]
        y = dataset.iloc[:, 8]
        X = X.values
        y = y.values
        pass

    elif dataset_name == 'connect-4.data':
        dataset = pd.read_csv(file_path, sep=',', skiprows=0)
        X = dataset.iloc[:, 0:42]
        y = dataset.iloc[:, 42]
        X = X.replace({"b": 0, "x": 1, "o": 2})
        y = y.replace({"win": 0, "loss": 1, "draw": 1})
        X = X.values
        y = y.values
        pass

    elif dataset_name == 'spambase.data':
        dataset = pd.read_csv(file_path, delimiter=",")
        X = dataset.iloc[:, 0:57]
        y = dataset.iloc[:, 57]
        X = X.values
        y = y.values
        pass


    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    return X, y, num_features, num_classes
