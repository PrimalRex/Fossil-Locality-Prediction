import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner as kt
from sklearn.utils import compute_class_weight
from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing, displayMetricsAgainstRandomGuessingMultiClass


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

    classWeights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(yTrain),
        y=yTrain
    )
    classWeights = dict(enumerate(classWeights))

    # Start the search
    print("=========================== STARTING BAYESIAN HYPERPARAM SEARCH ===========================")
    tuner.search(xTrain, yTrain, epochs=maxEpochs, validation_data=(xVal, yVal), callbacks=[earlyStopping, lrScheduler], class_weight=classWeights)

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

    print("=========================== FINISHED BAYESIAN HYPERPARAM SEARCH ===========================")

# Hyperband Optimisation for Binary Classification
def HyperbandBinaryOptim(skeleton, xTrain, yTrain, xVal, yVal, xTest, yTest, maxEpochs=50, binaryThreshold=0.90, batchSize=32, prefix="HYPERBAND_BINARY"):
    # Initialise the tuner
    tuner = kt.Hyperband(
        skeleton,
        objective="val_accuracy",
        max_epochs=maxEpochs,
        executions_per_trial=1,
        directory="kerastuner",
        project_name=prefix
    )
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    classWeights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(yTrain),
        y=yTrain
    )
    classWeights = dict(enumerate(classWeights))

    # Start the search
    print("=========================== STARTING HYPERBAND HYPERPARAM SEARCH ===========================")
    tuner.search(xTrain, yTrain, epochs=maxEpochs, validation_data=(xVal, yVal), callbacks=[earlyStopping, lrScheduler], class_weight=classWeights, batch_size=batchSize)

    # Evaluate the best performing model on the test set
    testPredictions = tuner.get_best_models(num_models=1)[0].predict(xTest).flatten()
    testBinaryPredictions = (testPredictions >= binaryThreshold).astype(int)
    displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictions, testBinaryPredictions, prefix, False)

    # Get the best hyperparameters
    bestParams = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print the best values
    print("Best Hyperband Hyperparameters:")
    for param, value in bestParams.values.items():
        print(f"{param}: {value}")

    print("=========================== FINISHED HYPERBAND HYPERPARAM SEARCH ===========================")

# Hyperband Optimisation for Categorical Classification
def HyperbandCategoricalOptim(skeleton, xTrain, yTrain, xVal, yVal, xTest, yTest, maxEpochs=50, batchSize=32, prefix="HYPERBAND_CATEGORICAL"):
    # Initialise the tuner
    tuner = kt.Hyperband(
        skeleton,
        objective="val_accuracy",
        max_epochs=maxEpochs,
        executions_per_trial=1,
        directory="kerastuner",
        project_name=prefix
    )
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    classWeights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(np.argmax(yTrain, axis=1)),
        y=np.argmax(yTrain, axis=1)
    )
    classWeights = dict(enumerate(classWeights))

    # Start the search
    print("=========================== STARTING HYPERBAND HYPERPARAM SEARCH ===========================")
    tuner.search(xTrain, yTrain, epochs=maxEpochs, validation_data=(xVal, yVal), callbacks=[earlyStopping, lrScheduler], class_weight=classWeights, batch_size=batchSize)

    # Evaluate the best performing model on the test set
    testPredictions = tuner.get_best_models(num_models=1)[0].predict(xTest)
    testClassPredictions = np.argmax(testPredictions, axis=1)
    displayMetricsAgainstRandomGuessingMultiClass(np.argmax(yTrain, axis=1), np.argmax(yTest, axis=1), testPredictions, testClassPredictions)

    # Get the best hyperparameters
    bestParams = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print the best values
    print("Best Hyperband Hyperparameters:")
    for param, value in bestParams.values.items():
        print(f"{param}: {value}")

    print("=========================== FINISHED HYPERBAND HYPERPARAM SEARCH ===========================")

# Multihead equivalent of Hyperband Optimisation for Categorical Classification
def HyperbandMultiHeadCategoricalOptim(skeleton, xTrainClimate, xTrainHydro, xTrainErosion, yTrain, xValClimate, xValHydro,
                                       xValErosion, yVal, xTestClimate, xTestHydro, xTestErosion, yTest, maxEpochs=50, batchSize=32, prefix="HYPERBAND_CATEGORICAL"):
    # Initialise the tuner
    tuner = kt.Hyperband(
        skeleton,
        objective="val_accuracy",
        max_epochs=maxEpochs,
        executions_per_trial=1,
        directory="kerastuner",
        project_name=prefix
    )
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    classWeights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(np.argmax(yTrain, axis=1)),
        y=np.argmax(yTrain, axis=1)
    )
    classWeights = dict(enumerate(classWeights))

    # Start the search
    print("=========================== STARTING HYPERBAND HYPERPARAM SEARCH ===========================")
    tuner.search(
        [xTrainClimate, xTrainHydro, xTrainErosion], yTrain,
        epochs=maxEpochs,
        validation_data=([xValClimate, xValHydro, xValErosion], yVal),
        callbacks=[earlyStopping, lrScheduler],
        class_weight=classWeights,
        batch_size=batchSize
    )

    # Evaluate the best performing model on the test set
    testPredictions = tuner.get_best_models(num_models=1)[0].predict([xTestClimate, xTestHydro, xTestErosion])
    testClassPredictions = np.argmax(testPredictions, axis=1)
    displayMetricsAgainstRandomGuessingMultiClass(np.argmax(yTrain, axis=1), np.argmax(yTest, axis=1), testPredictions, testClassPredictions)

    # Get the best hyperparameters
    bestParams = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print the best values
    print("Best Hyperband Hyperparameters:")
    for param, value in bestParams.values.items():
        print(f"{param}: {value}")

    print("=========================== FINISHED HYPERBAND HYPERPARAM SEARCH ===========================")