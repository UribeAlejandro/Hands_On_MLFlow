from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from utils.etl import extract_data, transform_data
from utils.ExperimentLogger import ExperimentLogger
from utils.training import experiment_auto_logger, experiment_low_level

RANDOM_STATE = 42


def training_loop() -> None:

    X, y = extract_data()
    X, y = transform_data(X, y)

    X_train, X_test, y_train, y_test = (
        X[:60000],
        X[60000:],
        y[:60000],
        y[60000:],
    )

    model = SGDClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    logger = ExperimentLogger(X_train, X_test, y_train, y_test, model)
    logger.log_experiment()

    experiment_low_level(X_train, X_test, y_train, y_test, model)
    experiment_auto_logger(X_train, X_test, y_train, y_test, model)


if __name__ == "__main__":
    training_loop()
