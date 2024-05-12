import pandas as pd


LOWER_PRICE_THRESHOLD = 500
UPPER_PRICE_THRESHOLD = 5_000_000


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    new_df = df.copy()
    new_df = new_df.drop(columns, axis=1)
    return new_df


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df = new_df.fillna("N/A")
    return new_df


def drop_if(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    # drop if no price, manufacturer and model
    new_df = new_df.dropna(
        subset=["price", "manufacturer", "model"]
    )  # Drop rows with NaN in "price" column
    new_df = new_df[
        new_df["price"] > LOWER_PRICE_THRESHOLD
    ]  # Drop rows with price <= 0
    new_df = new_df[
        new_df["price"] < UPPER_PRICE_THRESHOLD
    ]  # Drop rows with price <= 0
    return new_df


def add_column(df: pd.DataFrame) -> pd.DataFrame:
    pass


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

try:
    df = pd.read_csv("data/vehicles.csv")
    print("old len: ", len(df))
    # preprocessing
    new_df = drop_columns(df, columns_to_drop)
    new_df = drop_if(new_df)
    new_df = fill_na(new_df)
    print("new len: ", len(new_df))
    new_df.to_csv("data/preproc-vehicles.csv")
    new_df.head(100).to_csv("data/preproc-vehicles100.csv")
    print("Successfully saved preprocessed data")
except:
    print("Preprocessing failed")
