from load_data import load_preprocessed, load_original
import numpy as np
from pandas import DataFrame


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


original_df = load_original()["price"]
preprocessed_df = load_preprocessed()["price"]

print("Original data stats:")
original_stats = calc_stats(original_df)

print("-----------")

print("Preprocessed data stats:")
preprocessed_stats = calc_stats(preprocessed_df)
