from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist


def extract_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    mnist_data = mnist
    (training_images, training_labels), (
        test_images,
        test_labels,
    ) = mnist_data.load_data()

    return training_images, training_labels, test_images, test_labels


def transform_data(
    training_images: np.ndarray,
    test_images: np.ndarray,
    pca_bool: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    training_images, test_images = training_images.astype(
        "float32"
    ), test_images.astype("float32")

    training_images = training_images / 255.0 - 0.5
    test_images = test_images / 255.0 - 0.5

    if pca_bool:
        pca = PCA(0.95)

        training_images = training_images.reshape(60000, 784)
        training_images = pca.fit_transform(training_images)
    else:
        training_images = training_images.reshape(60000, 28, 28, 1)
        test_images = test_images.reshape(10000, 28, 28, 1)

    return training_images, test_images
