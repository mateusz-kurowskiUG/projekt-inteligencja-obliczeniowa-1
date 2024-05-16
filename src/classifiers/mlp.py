from sklearn.neural_network import MLPClassifier
from src.utils.preprocess import load_normalized
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from rich import print
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from joblib import dump
from seaborn import heatmap
import numpy as np

RANDOM_STATE = 288490


param_grid = {
    "solver": ["adam", "sgd"],
    "learning_rate_init": [0.01, 0.001],
    "hidden_layer_sizes": [
        (6,),
        # (8, 16),
        # (8, 16, 8),
        # (
        #     8,
        #     16,
        #     32,
        #     2,
        # ),
        # (8,),
        # (8, 32, 16, 2),
        # (
        #     8,
        #     16,
        #     4,
        #     2,
        # ),
    ],
    # "max_iter": [5000],
    "activation": ["relu", "tanh"],
    "batch_size": [64],
    "learning_rate": ["constant", "adaptive"],
}


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    return plt


def grid_search():
    data = load_normalized(part=0.1)
    data_X, data_y = data.drop("price", axis=1), data["price"]

    label_encoder = LabelEncoder()
    data_y = label_encoder.fit_transform(data_y)

    train_X, test_X, train_y, test_y = train_test_split(
        data_X, data_y, random_state=RANDOM_STATE, shuffle=True, test_size=0.2
    )
    grid_search = GridSearchCV(
        MLPClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=10,
        verbose=True,
        pre_dispatch="2*n_jobs",
    )
    grid_search.fit(train_X, train_y)
    # Get the best parameters and best score
    best_est = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    pred_y = grid_search.predict(test_X)
    accuracy = accuracy_score(test_y, pred_y)
    report = classification_report(test_y, pred_y)
    print(report)
    dump(best_est, "./data/best-mlp.joblib")

    cm = confusion_matrix(test_y, pred_y)

    plt.figure(figsize=(14, 12))
    heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xticks(rotation=45)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Best MLP model \n Accuracy: {accuracy}")
    plt.savefig("./plots/MLP/best-MLP.png")
    plt.close()  # Close the current figure

    title = "Learning Curves (MLP)"
    curve_plot = plot_learning_curve(
        grid_search.best_estimator_, title, train_X, train_y, cv=10, n_jobs=-1
    )
    curve_plot.savefig("./plots/MLP/best-curve.png")


# def main():
#     data = load_normalized(part=0.1)
#     data_X, data_y = data.drop("price", axis=1), data["price"]

#     label_encoder = LabelEncoder()
#     data_y = label_encoder.fit_transform(data_y)

#     train_X, test_X, train_y, test_y = train_test_split(
#         data_X, data_y, random_state=RANDOM_STATE
#     )
#     for i, model_args in enumerate(params):

#         mlp = MLPClassifier(random_state=RANDOM_STATE, **model_args)

#         mlp.fit(train_X, train_y)
#         predictions = mlp.predict(test_X)

#         accuracy = accuracy_score(test_y, predictions)
#         rounded_accuracy = round(accuracy * 1000) / 10
#         print(f"Run {i}: Accuracy: {accuracy}%")
#         # Wygeneruj wykres dokładności dla każdej konfiguracji


if __name__ == "__main__":
    # main()
    grid_search()
