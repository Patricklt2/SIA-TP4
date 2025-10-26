import pandas as pd 
from sklearn.preprocessing import StandardScaler

def load_and_std_europa(path):
    df = pd.read_csv(path)

    country_labels = df['Country'].values
    numerics_vars = df.columns.drop('Country')
    numerics_data = df[numerics_vars].values

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerics_data)

    return scaled_data, country_labels, numerics_vars.tolist()