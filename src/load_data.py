import pandas as pd


def load_preprocessed(part=1):
    df = pd.read_csv("../data/preproc-vehicles.csv").sample(frac=part)
    return df


def load_original():
    df = pd.read_csv("../data/vehicles.csv").sample(frac=1)
    return df
