from src.utils.preprocess import load_preprocessed, load_original
import numpy as np
from pandas import DataFrame
from rich import print


def calc_stats(df: DataFrame):
    # Oblicz podstawowe statystyki opisowe
    mean = np.mean(df)  # Średnia
    median = np.median(df)  # Mediana
    std_dev = np.std(df)  # Odchylenie standardowe
    min_val = np.min(df)  # Wartość minimalna
    max_val = np.max(df)  # Wartość maksymalna
    quantiles = np.percentile(df, [25, 50, 75])  # Kwartyle

    print("Średnia:", mean)
    print("Mediana:", median)
    print("Odchylenie standardowe:", std_dev)
    print("Wartość minimalna:", min_val)
    print("Wartość maksymalna:", max_val)
    print("Kwartyle (25%, 50%, 75%):", quantiles)


original_df = load_original()["odometer"]
preprocessed_df = load_preprocessed()["odometer"]

print("Original data stats:")
calc_stats(original_df)

print("-----------")

print("Preprocessed data stats:")
calc_stats(preprocessed_df)

# min_max(preprocessed_df)
