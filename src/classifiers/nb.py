from src.utils.preprocess import load_normalized

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import (
    GaussianNB,
    CategoricalNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
)
from seaborn import heatmap
from rich import print
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from numpy import argmax

data = load_normalized()
data_X, data_y = data.drop(["price"], axis=1), data["price"]

label_encoder = LabelEncoder()
data_y = label_encoder.fit_transform(data_y)

train_X, test_X, train_y, test_y = train_test_split(
    data_X, data_y, shuffle=True, random_state=288490, test_size=0.2
)


classifiers = [GaussianNB, CategoricalNB, MultinomialNB, BernoulliNB, ComplementNB]
classifiers_names = [
    "GaussianNB",
    "CategoricalNB",
    "MultinomialNB",
    "BernoulliNB",
    "ComplementNB",
]
accuracy_history = []
for classifier, classifier_name in zip(classifiers, classifiers_names):

    nb_classifier = classifier()
    nb_classifier.fit(train_X, train_y)
    y_pred = nb_classifier.predict(test_X)

    accuracy = accuracy_score(test_y, y_pred)
    f1 = f1_score(y_pred, test_y, average="weighted")
    rounded_f1 = round(f1, 3)
    rounded_accuracy = round(accuracy, 3)
    accuracy_history.append(rounded_accuracy)
    print(f"Dokładność klasyfikacji {classifier_name}:", rounded_accuracy)
    print("F1 Score:", rounded_f1)

    cm = confusion_matrix(test_y, y_pred)
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
    plt.title(f"Confusion Matrix {classifier_name} Accuracy:{rounded_accuracy}")
    plt.savefig(f"./plots/NB/{classifier_name}-CM.png")
    # dump(classifier, f"./models/NB/{classifier_name}.joblib")
    print("----------")

best_accuracy_index = argmax(accuracy_history)
best_accuracy = accuracy_history[best_accuracy_index]
print(f"best accuracy: {best_accuracy} on index: {best_accuracy_index}")
