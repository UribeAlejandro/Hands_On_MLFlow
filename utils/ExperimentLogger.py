import json
import os
import pickle
from datetime import datetime as dt

from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, fbeta_score


class ExperimentLogger:
    def __init__(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
        y_train: Series,
        y_test: Series,
        model,
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.parent_folder = f"experiment/my_own_private_tracker/{dt.now()}"

    def log_experiment(self) -> None:

        try:
            os.makedirs(self.parent_folder)
        except Exception as e:
            print(e)

        self.log_metrics()
        self.log_parameters()
        self.store_model()

    def log_metrics(self):
        metrics = {
            "training": {
                "accuracy_score": accuracy_score(
                    self.y_train, self.model.predict(self.X_train)
                ),
                "f_beta_avg_macro": fbeta_score(
                    self.y_train,
                    self.model.predict(self.X_train),
                    beta=2,
                    average="macro",
                ),
            },
            "test": {
                "accuracy_score": accuracy_score(
                    self.y_test, self.model.predict(self.X_test)
                ),
                "f_beta_avg_macro": fbeta_score(
                    self.y_test,
                    self.model.predict(self.X_test),
                    beta=2,
                    average="macro",
                ),
            },
        }

        with open(f"{self.parent_folder}/metrics.json", "w") as f:
            json.dump(metrics, f)

    def log_parameters(self):
        params = self.model.get_params()
        with open(f"{self.parent_folder}/params.json", "w") as f:
            json.dump(params, f)

    def store_model(self):
        with open(f"{self.parent_folder}/model.pkl", "wb") as f:
            pickle.dump(self.model, f)
