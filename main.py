from tensorflow.config.experimental import list_physical_devices
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential

from utils.etl import extract_data, transform_data
from utils.training import experiment_auto_logger

RANDOM_STATE = 42

print("Num GPUs Available: ", len(list_physical_devices("GPU")))


class accuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.998:
            print("\nReached 99.8% accuracy so cancelling training!\n")
            self.model.stop_training = True


def training_loop() -> None:

    training_images, training_labels, test_images, test_labels = extract_data()
    training_images, test_images = transform_data(
        training_images, test_images, True
    )

    accCallback = accuracyCallback()

    val_images = training_images[50000:]
    val_labels = training_labels[50000:]

    training_images = training_images[:50000]
    training_labels = training_labels[:50000]

    model = Sequential(
        [
            Conv1D(64, (3), activation="relu", input_shape=(154, 1)),
            MaxPooling1D(2, 2),
            Conv1D(64, (3), activation="relu"),
            MaxPooling1D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(model.summary())

    experiment_auto_logger(
        training_images,
        val_images,
        test_images,
        training_labels,
        val_labels,
        test_labels,
        model,
        [accCallback],
    )


if __name__ == "__main__":
    training_loop()
