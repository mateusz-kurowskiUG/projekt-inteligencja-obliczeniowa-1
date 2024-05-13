from pandas import read_csv, DataFrame
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

LOWER_PRICE_THRESHOLD = 500
UPPER_PRICE_THRESHOLD = 5_000_000
CATEGORICAL_COLUMNS = [
    "region",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "VIN",
    "drive",
    "type",
    "paint_color",
    "posting_date",
    "description",
]
NUMERICAL_COLUMNS = ["year", "odometer"]


def load_normalized(part=1, return_X_y=False):
    df = read_csv("./data/normalized-vehicles.csv", index_col=None).sample(
        frac=part, random_state=288490
    )
    if not return_X_y:
        return df
    return df.drop("price", axis=1), df["price"]


def load_preprocessed(part=1, return_X_y=False, encode_X=False):
    df = read_csv("./data/preproc-vehicles.csv", index_col=None).sample(
        frac=part, random_state=288490
    )
    if not return_X_y and not encode_X:
        return df

    if return_X_y:
        X = df.drop("price", axis=1)
        y = df["price"]

    if encode_X:
        x_encoded = encode(X)
        return x_encoded, y
    return X, y


def load_original(return_X_y=False):
    df = read_csv("./data/vehicles.csv", index_col=None).sample(
        frac=1, random_state=288490
    )
    if return_X_y:
        return df.drop("price", axis=1), df["price"]
    return df


def encode(data):
    return OneHotEncoder().fit_transform(data)


def drop_columns(df: DataFrame, columns: list[str]) -> DataFrame:
    new_df = df.drop(columns, axis=1)
    return new_df


def fill_na(df: DataFrame) -> DataFrame:
    numerical_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    # Uzupełnij brakujące wartości w danych numerycznych medianą
    df[NUMERICAL_COLUMNS] = numerical_imputer.fit_transform(df[NUMERICAL_COLUMNS])

    # Uzupełnij brakujące wartości w danych kategorycznych najczęściej występującą wartością
    df[CATEGORICAL_COLUMNS] = categorical_imputer.fit_transform(df[CATEGORICAL_COLUMNS])
    return new_df


def drop_if(df: DataFrame) -> DataFrame:
    new_df = df.dropna(
        subset=["price", "manufacturer", "model"]
    )  # Drop rows with NaN in "price" column
    new_df = new_df[
        new_df["price"] > LOWER_PRICE_THRESHOLD
    ]  # Drop rows with price <= 0
    new_df = new_df[
        new_df["price"] < UPPER_PRICE_THRESHOLD
    ]  # Drop rows with price <= 0
    return new_df


def normalize(df: DataFrame) -> DataFrame:
    # Wykonaj kodowanie kategorycznych danych za pomocą LabelEncoder
    label_encoder = LabelEncoder()
    for col in CATEGORICAL_COLUMNS:
        df[col] = label_encoder.fit_transform(df[col])

    # Wykonaj skalowanie danych numerycznych za pomocą MinMaxScaler
    scaler = MinMaxScaler()
    df[NUMERICAL_COLUMNS] = scaler.fit_transform(df[NUMERICAL_COLUMNS])

    return df


columns_to_drop = [
    "id",
    "url",
    "region_url",
    "lat",
    "long",
    "county",
    "state",
    "image_url",
    "size",
]
if __name__ == "__main__":
    try:
        df = load_original()
        print("old len: ", len(df))
        # preprocessing
        new_df = drop_columns(df, columns_to_drop)
        new_df = drop_if(new_df)
        new_df = fill_na(new_df)
        print("new len: ", len(new_df))
        new_df.to_csv("./data/preproc-vehicles.csv", index=False)
        new_df.head(10).to_csv("./data/preproc-vehicles10.csv", index=False)
        print("Successfully saved preprocessed data")
        normalized = normalize(new_df)
        normalized.to_csv("./data/normalized-vehicles.csv", index=False)
        print("Successfully saved normalized data")

    except Exception as exc:
        print("Preprocessing failed")
        print(exc)

# zero

# n=1
# Dokładność klasyfikacji k-NN k=1: 0.37496564990381975
# QSocketNotifier: Can only be used with threads started with QThread
# ----------
# n=3
# Dokładność klasyfikacji k-NN k=3: 0.28084638636988185
# ----------
# n=5
# Dokładność klasyfikacji k-NN k=5: 0.23605386095081066

# bez region
# n=1
# Dokładność klasyfikacji k-NN k=1: 0.37304204451772466
# QSocketNotifier: Can only be used with threads started with QThread
# ----------
# n=3
# Dokładność klasyfikacji k-NN k=3: 0.2782357790601814
# ----------
# n=5
# Dokładność klasyfikacji k-NN k=5: 0.23234405056334156

# bez description
# n=1
# Dokładność klasyfikacji k-NN k=1: 0.3741412475954933
# QSocketNotifier: Can only be used with threads started with QThread
# ----------
# n=3
# Dokładność klasyfikacji k-NN k=3: 0.28057158560043965
# ----------
# n=5
# Dokładność klasyfikacji k-NN k=5: 0.23660346248969497
# ----------
# best accuracy: 0.3741412475954933 on index: 0
