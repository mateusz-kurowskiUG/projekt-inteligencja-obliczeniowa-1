from src.utils.preprocess import load_normalized, load_preprocessed
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from src.utils.check_prediction import custom_accuracy_score
import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

sample = 0.5
data_X, data_y = load_normalized(part=sample, return_X_y=True)
# data_X, data_y = load_preprocessed(part=1, return_X_y=True, encode_X=True)

train_X, test_X, train_y, test_y = train_test_split(
    data_X, data_y, shuffle=True, random_state=288490, test_size=0.2
)
truthy_test_y = [True for i in test_y]
cm_labels = [True, False]
dtc = DecisionTreeClassifier()
classifier_name = "DTC"
dtc.fit(train_X, train_y)
y_pred = dtc.predict(test_X)
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
plt.title(f"Confusion Matrix {classifier_name} sample ={sample*100}%")
plt.savefig(f"./plots/DT/{classifier_name}-CM.png")
