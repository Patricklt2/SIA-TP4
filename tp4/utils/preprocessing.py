import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_std_europa(path=None):
    """
    Load europe.csv and return (data_std, country_labels, feature_names).
    If path is None or not found, try to locate the data file relative to this module.
    """
    if path is None or not os.path.exists(path):
        base_dir = os.path.dirname(__file__)
        candidate = os.path.normpath(os.path.join(base_dir, "..", "data", "europe.csv"))
        if os.path.exists(candidate):
            path = candidate
        else:
            raise FileNotFoundError(f"Could not find 'europe.csv' at {path!r} or {candidate!r}")

    df = pd.read_csv(path)
    if 'Country' not in df.columns:
        raise ValueError("Expected column 'Country' in europe.csv")

    country_labels = df['Country'].values
    numerics_vars = df.columns.drop('Country')
    numerics_data = df[numerics_vars].values.astype(float)

    scaler = StandardScaler()
    numerics_data_std = scaler.fit_transform(numerics_data)

    return numerics_data_std, country_labels, list(numerics_vars)