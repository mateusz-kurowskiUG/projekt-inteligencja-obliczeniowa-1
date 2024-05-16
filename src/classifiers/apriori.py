from src.utils.preprocess import load_preprocessed
from pandas import DataFrame, get_dummies, Series
from mlxtend.frequent_patterns import apriori, association_rules

from rich import print


def classify_year(year):
    if year < 1921:
        return "1901-1920"
    elif year < 1941:
        return "1921-1940"
    elif year < 1961:
        return "1941-1960"
    elif year < 1981:
        return "1961-1980"
    elif year < 2001:
        return "1981-2000"
    elif year < 2021:
        return "2001-2020"
    else:
        return "2021-now"


def classify_odometer(odometer):
    if odometer < 50_000:
        return "<0, 50k)"
    elif odometer < 100_000:
        return "<50k, 100k)"
    elif odometer < 200_000:
        return "<100k, 200k)"
    elif odometer < 500_000:
        return "<200k, 500k)"
    elif odometer < 1_000_000:
        return "<500k, 1mil)"
    else:
        return "<1mil, inf)"


def set_years(col: Series):
    col = col.apply(classify_year)
    return col


def set_odometers(col: Series):
    col = col.apply(classify_odometer)
    return col


def change_year_and_odometer(data: DataFrame):
    data["year"] = set_years(data["year"])
    data["odometer"] = set_odometers(data["odometer"])
    return data


def save_results(results: DataFrame):
    results.to_csv("./data/assoc_rules.csv")


if __name__ == "__main__":
    # Load and preprocess your data
    data = load_preprocessed()

    # Apply changes to 'year' and 'odometer' columns
    data_encoded = change_year_and_odometer(data)

    # One-hot encode the data
    data_encoded = get_dummies(data_encoded)

    # Generate frequent itemsets using Apriori algorithm
    frq_items = apriori(data_encoded, use_colnames=True, min_support=0.03, max_len=2)

    # Generate association rules
    rules = association_rules(frq_items, metric="lift", min_threshold=1)

    # Filter rules where 'title_status_clean' is not in consequents
    rules = rules[~rules["consequents"].apply(lambda x: "title_status_clean" in x)]

    # Sort and print the association rules
    rules = rules.sort_values(["confidence", "lift"], ascending=[False, False])

    # Print rules
    rules = rules.head(50)

    # Save results
    save_results(rules)
    print(rules)
