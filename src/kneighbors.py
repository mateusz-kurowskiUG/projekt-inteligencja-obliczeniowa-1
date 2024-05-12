from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from load_data import load_preprocessed
from joblib import dump
from rich import print


def predict(true_pred, pred):
    if pred >= 0.9 * true_pred and pred <= 1.1 * true_pred:
        return True
    else:
        return False


def custom_accuracy_score(y_test, y_pred) -> float:
    counter = 0
    results = []
    total_len = len(y_pred)
    for true_pred, pred in zip(y_test, y_pred):
        pred_valid = predict(true_pred, pred)
        if pred_valid:
            counter += 1
        results.append(pred_valid)
    return float(counter / total_len), results


def compare_predictions():
    pass


print("Loading data")
df = load_preprocessed(0.1)
X = df.drop("price", axis=1)
y = df["price"]

print("encodujemy")

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Podziel dane na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=288490, shuffle=True
)

truthy_test_y = [True for el in y_test]
cm_labels = [True, False]

neighbors_n_array = [3, 5, 7]
for n in neighbors_n_array:
    classifier = KNeighborsClassifier(n_neighbors=n)
    print(f"n={n}")
    print("fitujemy")
    classifier.fit(X_train, y_train)
    print("predictujemy")
    y_pred = classifier.predict(X_test)

    print("accuracy mierzymy")
    accuracy, real_predictions = custom_accuracy_score(y_test, y_pred)
    print(f"Dokładność klasyfikacji k-NN k={n}:", accuracy)

    print(truthy_test_y)
    print(real_predictions)
    # cm
    cm = confusion_matrix(truthy_test_y, real_predictions, labels=cm_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Real", "Predicted"]
    )
    disp.figure_.savefig(f"../plots/KNN-{n}-CM.png")
    dump(classifier, f"../models/knn-{n}.joblib")
    print("----------")
# Traceback (most recent call last):
#   File "/home/ubuntu/studia/inteligencja-proj1/src/kneighbors.py", line 64, in <module>
#     cm = confusion_matrix(truthy_test_y, real_predictions, labels=labels)
#   File "/home/ubuntu/.cache/pypoetry/virtualenvs/inteligencja-proj1-g7Dqjw1q-py3.10/lib/python3.10/site-packages/sklearn/utils/_param_validation.py", line 203, in wrapper
#     validate_parameter_constraints(
#   File "/home/ubuntu/.cache/pypoetry/virtualenvs/inteligencja-proj1-g7Dqjw1q-py3.10/lib/python3.10/site-packages/sklearn/utils/_param_validation.py", line 95, in validate_parameter_constraints
#     raise InvalidParameterError(
# sklearn.utils._param_validation.InvalidParameterError: The 'y_pred' parameter of confusion_matrix must be an array-like. Got True instead.
