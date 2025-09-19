# src/data_utils.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_csv(path)
    # coerce numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def split_target(df, target='MEDV'):
    if target not in df.columns:
        raise ValueError(f"{target} not in dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def impute_numeric(X, strategy='median'):
    nums = X.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy=strategy)
    X_copy = X.copy()
    X_copy[nums] = imputer.fit_transform(X_copy[nums])
    return X_copy, imputer
