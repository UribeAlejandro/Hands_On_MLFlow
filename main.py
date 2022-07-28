from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.ExperimentLogger import ExperimentLogger

__RANDOM_STATE = 42


def training_loop():

    X, y = load_iris(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        shuffle=True,
        test_size=0.3,
        random_state=__RANDOM_STATE,
    )

    model = DecisionTreeClassifier(random_state=__RANDOM_STATE)
    model.fit(X_train, y_train)

    logger = ExperimentLogger(X_train, X_test, y_train, y_test, model)
    logger.log_experiment()

    # experiment_low_level(X_train, X_test, y_train, y_test, model)
    # experiment_auto_logger(X_train, X_test, y_train, y_test, model)


if __name__ == "__main__":
    training_loop()
