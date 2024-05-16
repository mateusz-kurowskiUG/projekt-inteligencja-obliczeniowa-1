from pandas import read_csv, DataFrame, qcut
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from rich import print
import numpy as np

LOWER_PRICE_THRESHOLD = 500
UPPER_PRICE_THRESHOLD = 5_000_000


def return_categorical_and_numerical_cols(df: DataFrame):
    cat_cols = df.columns[df.dtypes == "object"]
    num_cols = df.columns[df.dtypes != "object"]
    return cat_cols, num_cols


def load_normalized(part=1):
    return read_csv("./data/normalized-vehicles.csv", index_col=None).sample(
        frac=part, random_state=288490
    )


def load_preprocessed(part=1):
    return read_csv("./data/preproc-vehicles.csv", index_col=None).sample(
        frac=part, random_state=288490
    )


def load_original():
    return read_csv("./data/vehicles.csv").sample(frac=1, random_state=288490)


def drop_columns(df: DataFrame, columns: list[str]) -> DataFrame:
    new_df = df.drop(columns, axis=1)
    return new_df


def fill_na(df: DataFrame) -> DataFrame:
    new_df = df.copy()
    cat_cols, num_cols = return_categorical_and_numerical_cols(df)

    numerical_imputer = SimpleImputer(
        strategy="median",
    )
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    # Uzupełnij brakujące wartości w danych numerycznych medianą
    new_df[num_cols] = numerical_imputer.fit_transform(df[num_cols])

    # Uzupełnij brakujące wartości w danych kategorycznych najczęściej występującą wartością
    new_df[cat_cols] = categorical_imputer.fit_transform(df[cat_cols])
    return new_df


def drop_if(df: DataFrame) -> DataFrame:
    new_df = df.drop("county", axis=1)
    # drop if more than 20% N/A in column
    some_null_cols = df.columns[df.isnull().mean() > 0.2]
    new_df = df.drop(columns=some_null_cols, axis=1)

    new_df = new_df[new_df["year"] < 2024]  # Drop rows with price <= 0
    new_df = new_df[new_df["year"] > 1900]  # Drop rows with price <= 0
    new_df = new_df[
        new_df["price"] > LOWER_PRICE_THRESHOLD
    ]  # Drop rows with price <= 0
    new_df = new_df[
        new_df["price"] < UPPER_PRICE_THRESHOLD
    ]  # Drop rows with price <= 0
    return new_df


def normalize(df: DataFrame) -> DataFrame:
    cat_cols, num_cols = return_categorical_and_numerical_cols(df)
    price_index = np.where(cat_cols == "price")
    cat_cols = cat_cols.delete(price_index)
    # Wykonaj kodowanie kategorycznych danych za pomocą LabelEncoder
    label_encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Wykonaj skalowanie danych numerycznych za pomocą MinMaxScaler
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def get_intervals(df: DataFrame) -> list[tuple[int, int]]:
    X = df["price"]  # Column of data
    num_intervals = 10  # Number of intervals
    intervals = qcut(X, q=num_intervals)
    my_intervals = []
    # Retrieving the right side of each interval
    for interval in intervals.unique():
        my_intervals.append((interval.left, interval.right))
    my_intervals.sort()  # Sorting the intervals
    return my_intervals


def get_class(price: int, intervals: list[tuple[int, int]]) -> str:
    for left, right in intervals:
        if price < right:
            left_k = round(left / 1000, 1)
            right_k = round(right / 1000, 1)
            return f"({left_k}k , {right_k}k>"
    # If price is greater than all intervals, return the last interval
    last = round(intervals[-1][1] / 1000, 1)
    return f"({last}k+"


def set_classes(df: DataFrame) -> DataFrame:
    new_df = df.copy()
    intervals = get_intervals(new_df)  # Get intervals from new_df
    new_df["price"] = new_df["price"].apply(get_class, intervals=intervals)
    return new_df


def set_easy_classes(df: DataFrame) -> DataFrame:
    median = df["price"].median()  # Calculate the median price
    # Use the apply method to generate labels based on the median price
    df["price"] = df["price"].apply(lambda x: "expensive" if x > median else "cheap")

    return df


def fix_types(df: DataFrame) -> DataFrame:
    new_df = df.copy()
    new_df["year"] = new_df["year"].astype(int)
    new_df["odometer"] = new_df["odometer"].astype(int)
    new_df["price"] = new_df["price"].astype(int)
    return new_df


def load_easy(part=1):
    return read_csv("./data/normalized3-easy-vehicles.csv")


columns_to_drop = [
    "id",
    "url",
    "region_url",
    "VIN",
    "image_url",
    "description",
    "state",
    "lat",
    "long",
    "posting_date",
    "title_status",
    "region",
]

if __name__ == "__main__":
    try:
        df = load_original()
        df.head(1).to_csv("./data/vehicles-1.csv", index=False)
        print("old len: ", len(df))
        # preprocessing

        new_df = drop_columns(df, columns_to_drop)

        new_df = drop_if(new_df)
        new_df = fill_na(new_df)
        new_df = fix_types(new_df)
        new_df = set_classes(new_df)  # Pass new_df to set_classes

        new_df2 = drop_if(df)
        new_df2 = fill_na(new_df2)
        new_df2 = fix_types(new_df2)

        new_df3 = new_df2.copy().drop(
            [
                "id",
                "url",
                "region",
                "region_url",
                "title_status",
                "image_url",
                "description",
                "state",
                "lat",
                "long",
                "posting_date",
            ],
            axis=1,
        )

        new_df2 = set_classes(new_df2)  # Pass new_df to set_classes
        new_df3 = set_easy_classes(new_df3)
        print(new_df3["price"].unique())

        print("new len: ", len(new_df))
        new_df.to_csv("./data/preproc-vehicles.csv", index=False)
        new_df.head(10).to_csv("./data/preproc-vehicles10.csv", index=False)
        print("Successfully saved preprocessed data")

        normalized = normalize(new_df)
        normalized2 = normalize(new_df2)
        normalized3 = normalize(new_df3)

        normalized.to_csv("./data/normalized-vehicles.csv", index=False)
        normalized2.to_csv("./data/normalized2-full-vehicles.csv", index=False)
        normalized3.to_csv("./data/normalized3-easy-vehicles.csv", index=False)
        print("Successfully saved normalized data")

    except Exception as exc:
        print("Preprocessing failed")
        print(exc)
