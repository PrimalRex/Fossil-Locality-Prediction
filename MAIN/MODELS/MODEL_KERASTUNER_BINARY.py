import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner as kt
from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing

# MAIN ------------------------------------------------------------------------

# This script handles the boilerplate for different kerastuner methods for hyperparameter optimisation

# Bayesian Optimisation for Binary Classification
def BayesianBinaryOptim(skeleton, xTrain, yTrain, xVal, yVal, xTest, yTest, maxTrials=15, binaryThreshold=0.90, maxEpochs=20, prefix="BAYESIAN_BINARY"):
    # Initialise the tuner
    tuner = kt.BayesianOptimization(
        skeleton,
        objective="val_accuracy",
        max_trials=maxTrials,
        executions_per_trial=1,
        directory="kerastuner",
        project_name=prefix
    )
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    # STart the search
    print("=========================== STARTING HYPERPARAM SEARCH ===========================")
    tuner.search(xTrain, yTrain, epochs=maxEpochs, validation_data=(xVal, yVal), callbacks=[earlyStopping, lrScheduler])

    # Evaluate the best performing model on the test set
    testPredictions = tuner.get_best_models(num_models=1)[0].predict(xTest).flatten()
    testBinaryPredictions = (testPredictions >= binaryThreshold).astype(int)
    displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictions, testBinaryPredictions, prefix, False)

    # Get the best hyperparameters
    bestParams = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print the best values
    print("Best Bayesian Hyperparameters:")
    for param, value in bestParams.values.items():
        print(f"{param}: {value}")

    print("=========================== FINISHED HYPERPARAM SEARCH ===========================")