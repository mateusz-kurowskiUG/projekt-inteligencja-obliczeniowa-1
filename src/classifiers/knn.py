from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from src.utils.preprocess import load_normalized
from rich import print
import matplotlib.pyplot as plt
from seaborn import heatmap, lineplot
from numpy import argmax, mean
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

data = load_normalized()

data_X, data_y = data.drop(["price"], axis=1), data["price"]

label_encoder = LabelEncoder()
data_y_encoded = label_encoder.fit_transform(data_y)

# Podziel dane na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.2, random_state=288490, shuffle=True
)


k_values = [i for i in range(1, 31, 2)]
scores = []
accuracy_history = []

for k in k_values:
    classifier = KNeighborsClassifier(n_neighbors=k)
    print(f"k={k}")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    score = cross_val_score(classifier, data_X, data_y, cv=5)
    scores.append(mean(score))

    rounded_accuracy = round(accuracy, 3)
    print(f"Dokładność klasyfikacji k-NN k={k}:", rounded_accuracy)
    accuracy_history.append(accuracy)
    # cm
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
        f"Confusion Matrix KNN k={k} \n Accuracy {round(rounded_accuracy*1000)/10}%"
    )
    plt.savefig(f"./plots/KNN/KNN-{k}-CM.png")
    plt.close()  # Close the current figure
    # print("----------")


best_accuracy_index = argmax(accuracy_history)
best_accuracy = accuracy_history[best_accuracy_index]
print(f"best accuracy: {best_accuracy} on index: {best_accuracy_index}")

## best k

lineplot(x=k_values, y=scores, marker="o")
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.savefig("./plots/KNN/best-k.png")
