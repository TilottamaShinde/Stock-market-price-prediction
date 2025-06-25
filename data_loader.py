import pandas as pd

def load_data(path = "raw_data.csv"):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df
