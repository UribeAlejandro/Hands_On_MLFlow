import mlflow


def experiment_auto_logger(
    X_train, X_val, X_test, y_train, y_val, y_test, model, callbacks
) -> None:

    mlflow.tensorflow.autolog()

    with mlflow.start_run(experiment_id="3") as run:
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            epochs=20,
        )
        test_eval = model.evaluate(X_test, y_test)
        mlflow.log_metric("Test_eval_loss", test_eval[0])
        mlflow.log_metric("Test_eval_accuracy", test_eval[1])

    mlflow.end_run()
