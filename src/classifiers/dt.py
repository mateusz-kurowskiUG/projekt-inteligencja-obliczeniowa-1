from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
from src.utils.preprocess import load_normalized
from rich import print
import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.preprocessing import LabelEncoder


data = load_normalized()

data_X, data_y = data.drop("price", axis=1), data["price"]

label_encoder = LabelEncoder()
data_y = label_encoder.fit_transform(data_y)

X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.2, random_state=288490, shuffle=True
)

trees = [
    {
        "criterion": "gini",
        "splitter": "best",
        "max_depth": None,
    },
    {
        "criterion": "gini",
        "splitter": "best",
        "max_depth": None,
    },
    {
        "criterion": "entropy",
        "splitter": "best",
        "max_depth": None,
    },
]


if __name__ == "__main__":
    for i, tree_params in enumerate(trees):
        criterion = tree_params["criterion"]
        splitter = tree_params["splitter"]
        max_depth = tree_params["max_depth"]

        dt_classifier = DecisionTreeClassifier(
            criterion=criterion, splitter=splitter, max_depth=max_depth
        )
        dt_classifier.fit(X_train, y_train)
        y_pred = dt_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        rounded_accuracy = round(accuracy, 3)
        print(f"Dokładność klasyfikacji DT no. {i} :", rounded_accuracy)

        tree_params["accuracy"] = accuracy

        cm = confusion_matrix(y_test, y_pred)

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
        plt.title(
            f"Decision Tree no. {i} \n criterion = {tree_params['criterion']} splitter = {tree_params['splitter']} \n max_depth = {tree_params['max_depth']} \n Accuracy {round(rounded_accuracy*1000)/10}%"
        )
        plt.savefig(f"./plots/DT/DT-{i}-CM.png")
        plt.close()  # Close the current figure
        if max_depth is not None:
            fig = plt.figure(figsize=(25, 20))
            tree_plot = plot_tree(dt_classifier, filled=True)
            fig.savefig(f"./plots/DT/decistion_tree-{i}.svg")
