from src.utils.preprocess import load_preprocessed

import tensorflow as tf
from keras import callbacks, layers, Sequential, utils,regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pandas import concat, DataFrame
from rich import print
import matplotlib.pyplot as plt


def prepare_data_for_ml(data: DataFrame) -> tuple[DataFrame, DataFrame]:
    data_X, data_y = data.drop("price", axis=1), data["price"]

    # Pick columnXs
    categorical_X_cols = data_X.select_dtypes(include=["object"]).columns.tolist()
    numerical_X_cols = data_X.select_dtypes(include=["number"]).columns.tolist()

    # Encoding numerical
    mm_scaler = MinMaxScaler()
    numerical_data_scaled = mm_scaler.fit_transform(data[numerical_X_cols])
    numerical_scaled_df = DataFrame(
        numerical_data_scaled, columns=mm_scaler.get_feature_names_out(numerical_X_cols)
    )

    # Encoding categorical
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    categorical_data_encoded = one_hot_encoder.fit_transform(data[categorical_X_cols])
    categorical_encoded_df = DataFrame(
        categorical_data_encoded,
        columns=one_hot_encoder.get_feature_names_out(categorical_X_cols),
    )

    # concat categorical and numerical dataframes
    encoded_X = concat([categorical_encoded_df, numerical_scaled_df], axis=1)

    # encode targets
    encoded_y = one_hot_encoder.fit_transform(data_y.to_frame())
    # make targets df
    encoded_y_df = DataFrame(
        encoded_y, columns=one_hot_encoder.get_feature_names_out(["price"])
    )
    return encoded_X, encoded_y_df


def create_model(shape):
    model = Sequential(

        [
            layers.Dense(256, activation="tanh", name="layer1"),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(128, activation="tanh", name="layer2"),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(64, activation="tanh", name="layer3"),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(32, activation="tanh", name="layer4"),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(10, activation="softmax", name="output"),
        ],
    )
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def plot(history, name):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"./plots/MLP/accuracy-{name}.png")
    plt.close()
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"./plots/MLP/loss-{name}.png")
    plt.close()


if __name__ == "__main__":
    # get the data
    data = load_preprocessed(part=0.15)
    data_X, data_y = prepare_data_for_ml(data)
    train_X, test_val_X, train_y, test_val_y = train_test_split(
        data_X, data_y, random_state=288490, shuffle=True, test_size=0.3
    )
    test_X, val_X, test_y, val_y = train_test_split(
        test_val_X, test_val_y, random_state=288490, shuffle=True, test_size=0.5
    )
    i=18
    # define callbacks
    early_stopping = callbacks.EarlyStopping(monitor="val_loss", verbose=1, restore_best_weights=True, patience=20, min_delta=0.001,)
    history = callbacks.History()
    checkpoint = callbacks.ModelCheckpoint(
        f"./models/checkpoint-{i}.model.keras",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )
    my_callbacks = [history, checkpoint, early_stopping]

    # model
    shape_row, shape_col = data_X.shape
    model = create_model((shape_col,))
    model.fit(
        train_X,
        train_y,
        validation_data=(val_X, val_y),
        epochs=1000,
        batch_size=128,
        callbacks=my_callbacks,
    )
    test_loss, test_accuracy = model.evaluate(test_X, test_y, batch_size=32, verbose=1)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")
    # plot
    # summarize history for accuracy
    plot(history, f"{i}")
