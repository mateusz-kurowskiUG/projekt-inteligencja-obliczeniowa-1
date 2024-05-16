import torch
import torch.nn as nn
from src.utils.preprocess import load_normalized
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

data_X, data_y = load_normalized(part=0.01, return_X_y=True)

X_array = data_X.values
y_array = data_y.values

X_tensor = torch.tensor(X_array, dtype=torch.float32)
y_tensor = torch.tensor(y_array, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# Tworzenie DataLoadera
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False
)  # Nie mieszamy danych testowych


model = nn.Sequential(
    nn.Linear(15, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 1),  # Ostatnia warstwa bez aktywacji (regresja)
)


# Definiowanie funkcji straty i optymalizatora
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Trening modelu
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Ustawienie modelu w tryb treningowy
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(
            outputs, targets.long()
        )  # targets muszą być w formacie LongTensor
        loss.backward()
        optimizer.step()

    # Walidacja modelu na zbiorze testowym
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            total_loss += loss.item() * len(inputs)

        avg_loss = total_loss / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_loss:.4f}")
