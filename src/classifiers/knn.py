from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from src.utils.preprocess import load_normalized, load_preprocessed
from joblib import dump
from rich import print
import matplotlib.pyplot as plt
from seaborn import heatmap
from numpy import argmax
from src.utils.check_prediction import custom_accuracy_score
from sklearn.impute import SimpleImputer

print("Loading data")

# Label encoder, minmax variant:
# data_X, data_y = load_normalized(part=1, return_X_y=True)
data_X, data_y = load_preprocessed(part=1, return_X_y=True, encode_X=True)

# data_X, data_y = load_preprocessed(part=1, return_X_y=True, encode_X=True)
# Podziel dane na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.2, random_state=288490, shuffle=True
)


truthy_test_y = [True for i in y_test]
cm_labels = [True, False]

neighbors_n_array = [1, 3, 5]
accuracy_history = []
for n in neighbors_n_array:
    classifier = KNeighborsClassifier(n_neighbors=n)
    print(f"n={n}")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy, real_predictions = custom_accuracy_score(y_test, y_pred)
    print(f"Dokładność klasyfikacji k-NN k={n}:", accuracy)
    accuracy_history.append(accuracy)
    # cm
    cm = confusion_matrix(truthy_test_y, real_predictions, labels=cm_labels)

    # Sum along axis 0 to collapse the rows into a single row
    cm_1x2 = cm.sum(axis=0)

    # Plotting
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
    plt.title(f"Confusion Matrix KNN n={n}")
    plt.savefig(f"./plots/KNN/KNN-{n}-CM.png")
    dump(classifier, f"./models/KNN/knn-{n}.joblib")
    print("----------")

best_accuracy_index = argmax(accuracy_history)
best_accuracy = accuracy_history[best_accuracy_index]
print(f"best accuracy: {best_accuracy} on index: {best_accuracy_index}")


### normalizowane StandartScaler oraz LabelEncoder:

# n=1
# Dokładność klasyfikacji k-NN k=1: 0.3152687699241508
# QSocketNotifier: Can only be used with threads started with QThread
# ----------
# n=3
# Dokładność klasyfikacji k-NN k=3: 0.263383533032868
# ----------
# n=5
# Dokładność klasyfikacji k-NN k=5: 0.24793888094976366
# ----------
# best accuracy: 0.3152687699241508 on index: 0

# normalnizowane za pomocą OneHotEncoder:

### sam standartScaler
# Loading data
# n=1
# Dokładność klasyfikacji k-NN k=1: 0.3152687699241508
# QSocketNotifier: Can only be used with threads started with QThread
# ----------
# n=3
# Dokładność klasyfikacji k-NN k=3: 0.263383533032868
# ----------
# n=5
# Dokładność klasyfikacji k-NN k=5: 0.24793888094976366
# ----------
# best accuracy: 0.3152687699241508 on index: 0
