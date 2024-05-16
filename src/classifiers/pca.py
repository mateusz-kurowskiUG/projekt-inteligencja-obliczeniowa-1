from src.utils.preprocess import load_normalized, load_original
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from rich import print
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pca(X, y, nums, pn):
    var_ratio = []
    for num in nums:
        pca = PCA(n_components=num)
        pca.fit(X, y)
        var_ratio.append(
            np.sum(pca.explained_variance_ratio_)
        )  # Summing explained variance ratio for each number of components

    print(nums)
    print(var_ratio)
    plt.figure(figsize=(6, 6), dpi=150)
    plt.grid()
    plt.plot(nums, var_ratio, marker="o")
    plt.xlabel("n_components")
    plt.ylabel("Explained variance ratio")
    plt.title(f"n_components vs. Explained Variance Ratio \n {pn} deleting cols")
    plt.savefig(f"./plots/PCA/pca-{pn}.png")
    plt.close()


if __name__ == "__main__":

    # before normalization
    data = pd.read_csv("./data/normalized2-full-vehicles.csv")

    # Separate features and target variable
    data_X = data.drop("price", axis=1)
    data_y = data["price"]

    # Encode the target variable if it's categorical
    label_encoder = LabelEncoder()
    data_y_encoded = label_encoder.fit_transform(data_y)

    # Define the range of components for PCA
    nums = np.arange(1, 18)  # You can adjust this range as needed

    # Perform PCA
    pca(data_X, data_y_encoded, nums, "before")
    # after
    nums = np.arange(1, 7)  # Change from 6 to 7 to include all components

    data = load_normalized(part=1)
    data_X, data_y = data.drop("price", axis=1), data["price"]
    label_encoder = LabelEncoder()
    data_y_encoded = label_encoder.fit_transform(data_y)
    pca(data_X, data_y_encoded, nums, "after")
