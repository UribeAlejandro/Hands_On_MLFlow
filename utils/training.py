import os
from typing import Tuple

import mlflow
from sklearn.metrics import accuracy_score, fbeta_score


# 1
def experiment_low_level(X_train, X_test, y_train, y_test, model):
    acc_score_train = accuracy_score(y_train, model.predict(X_train))
    f_beta_train = fbeta_score(
        y_train, model.predict(X_train), beta=2, average="macro"
    )

    acc_score_test = accuracy_score(y_test, model.predict(X_test))
    f_beta_test = fbeta_score(
        y_test, model.predict(X_test), beta=2, average="macro"
    )

    with mlflow.start_run() as run:
        metrics = {
            "accuracy_score_train": accuracy_score(
                y_train, model.predict(X_train)
            ),
            "f_beta_avg_macro_train": fbeta_score(
                y_train, model.predict(X_train), beta=2, average="macro"
            ),
            "accuracy_score_test": accuracy_score(
                y_test, model.predict(X_test)
            ),
            "f_beta_avg_macro_test": fbeta_score(
                y_test, model.predict(X_test), beta=2, average="macro"
            ),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.get_params())

        mlflow.sklearn.log_model(model, "model")

    mlflow.end_run()


# 2
def experiment_auto_logger(X_train, X_test, y_train, y_test, model) -> None:

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        mlflow.sklearn.eval_and_log_metrics(
            model, X_test, y_test, prefix="val_"
        )

    mlflow.end_run()
