from src.utils.preprocess import load_normalized

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from seaborn import heatmap
from rich import print
import matplotlib.pyplot as plt
from src.utils.check_prediction import custom_accuracy_score
from sklearn.metrics import confusion_matrix
from joblib import dump

data_X, data_y = load_normalized(part=0.1, return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(
    data_X, data_y, shuffle=True, random_state=288490, test_size=0.2
)


classifiers = [GaussianNB]
classifiers_names = ["GaussianNB", "CategoricalNB", "CategoricalNB"]
truthy_test_y = [True for i in test_y]
cm_labels = [True, False]

for index, classifier in enumerate(classifiers):
    classifier_name = classifiers_names[index]
    nb_classifier = classifier()
    nb_classifier.fit(train_X, train_y)
    y_pred = nb_classifier.predict(test_X)
    accuracy, real_predictions = custom_accuracy_score(test_y, y_pred)
    print(f"Dokładność klasyfikacji {classifier_name}:", accuracy)
    cm = confusion_matrix(truthy_test_y, real_predictions, labels=cm_labels)
    cm_1x2 = cm.sum(axis=0)
    plt.figure(figsize=(8, 4))
    heatmap(
        cm_1x2.reshape(1, -1),
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["True", "False"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix {classifier_name}")
    plt.savefig(f"./plots/NB/{classifier_name}-CM.png")
    dump(classifier, f"./models/NB/{classifier_name}.joblib")
    print("----------")
