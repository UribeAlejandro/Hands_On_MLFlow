from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def extract_data() -> Tuple[pd.DataFrame, pd.Series]:

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

    return X, y


def transform_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:

    y = y.astype(np.uint8)

    return X, y
