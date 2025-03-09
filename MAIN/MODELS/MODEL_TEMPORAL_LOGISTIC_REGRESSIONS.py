import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight

from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing, displayMetricsAgainstRandomGuessingMultiClass
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This is a script to train logistic models for the temporal aspect of the predictions
# Models assume that the data splits have already been scaled appropriately before invocation

# This model trains on a single-timestep, usually the T1 timestep to appreciate whether temporal data is relevant for the classification problem
def logisticModel_T1(xTrain, yTrain, xTest, yTest, binaryThreshold=0.90, prefix="Test", multiClass=False):
    # Extract last timestep (T1)
    xTrainT1 = xTrain[:, -1, :]
    xTestT1 = xTest[:, -1, :]

    if not multiClass:
        # Compute class weights
        classWeights = compute_class_weight(class_weight="balanced",classes=np.unique(yTrain), y=yTrain)
        # Binary Logistic Regression
        logReg = LogisticRegression(max_iter=1000, class_weight=dict(enumerate(classWeights)))
        logReg.fit(xTrainT1, yTrain)

        testPredictions = logReg.predict_proba(xTestT1)[:, 1]
        testBinaryPredictions = (testPredictions >= binaryThreshold).astype(int)
        displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictions, testBinaryPredictions, f"{prefix} LOGISTIC T1", False)
    else:
        # Compute class weights
        classWeights = compute_class_weight(class_weight="balanced",classes=np.unique(np.argmax(yTrain, axis=1)),y=np.argmax(yTrain, axis=1))
        # MultiClass Logistic Regression
        logReg = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", class_weight=dict(enumerate(classWeights)))
        logReg.fit(xTrainT1, np.argmax(yTrain, axis=1))

        testPredictions = logReg.predict_proba(xTestT1)
        testClassPredictions = np.argmax(testPredictions, axis=1)
        displayMetricsAgainstRandomGuessingMultiClass(np.argmax(yTrain, axis=1), np.argmax(yTest, axis=1), testPredictions, testClassPredictions, f"{prefix} LOGISTIC T1", False)

# This model flattens all the timesteps into a single feature vector to appreciate temporal order, and if temporal importance is relevant for the classification problem
def logisticModel_FlatVector(xTrain, yTrain, xTest, yTest, binaryThreshold=0.90, prefix="Test", multiClass=False):
    xTrainFlatVector = xTrain.reshape(xTrain.shape[0], -1)
    xTestFlatVector = xTest.reshape(xTest.shape[0], -1)

    if not multiClass:
        # Compute class weights
        classWeights = compute_class_weight(class_weight="balanced",classes=np.unique(yTrain), y=yTrain)
        logRegFlatVector = LogisticRegression(max_iter=1000, class_weight=dict(enumerate(classWeights)))
        logRegFlatVector.fit(xTrainFlatVector, yTrain)

        testPredictionsFlatVector = logRegFlatVector.predict_proba(xTestFlatVector)[:, 1]
        testBinaryPredictionsFlatVector = (testPredictionsFlatVector >= binaryThreshold).astype(int)
        displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictionsFlatVector, testBinaryPredictionsFlatVector, f"{prefix} LOGISTIC Flat Vector", False)
    else:
        # Compute class weights
        classWeights = compute_class_weight(class_weight="balanced",classes=np.unique(np.argmax(yTrain, axis=1)),y=np.argmax(yTrain, axis=1))
        logRegFlatVector = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", class_weight=dict(enumerate(classWeights)))
        logRegFlatVector.fit(xTrainFlatVector, np.argmax(yTrain, axis=1))

        testPredictionsFlatVector = logRegFlatVector.predict_proba(xTestFlatVector)
        testClassPredictionsFlatVector = np.argmax(testPredictionsFlatVector, axis=1)
        displayMetricsAgainstRandomGuessingMultiClass(np.argmax(yTrain, axis=1), np.argmax(yTest, axis=1), testPredictionsFlatVector, testClassPredictionsFlatVector, f"{prefix} LOGISTIC Flat Vector", False)


# This model averages all the timesteps into a single timestep T(X) to appreciate if long-term data varies in comparison to short-term data
def logisticModel_MeanAverage(xTrain, yTrain, xTest, yTest, binaryThreshold=0.90, prefix="Test", multiClass=False):
    xTrainAvg = np.mean(xTrain, axis=1)
    xTestAvg = np.mean(xTest, axis=1)

    if not multiClass:
        # Compute class weights
        classWeights = compute_class_weight(class_weight="balanced",classes=np.unique(yTrain), y=yTrain)
        logRegAvg = LogisticRegression(max_iter=1000, class_weight=dict(enumerate(classWeights)))
        logRegAvg.fit(xTrainAvg, yTrain)

        testPredictionsAvg = logRegAvg.predict_proba(xTestAvg)[:, 1]
        testBinaryPredictionsAvg = (testPredictionsAvg >= binaryThreshold).astype(int)
        displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictionsAvg, testBinaryPredictionsAvg,f"{prefix} LOGISTIC Mean Average", False)
    else:
        # Compute class weights
        classWeights = compute_class_weight(class_weight="balanced",classes=np.unique(np.argmax(yTrain, axis=1)),y=np.argmax(yTrain, axis=1))
        logRegAvg = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", class_weight=dict(enumerate(classWeights)))
        logRegAvg.fit(xTrainAvg, np.argmax(yTrain, axis=1))

        testPredictionsAvg = logRegAvg.predict_proba(xTestAvg)
        testClassPredictionsAvg = np.argmax(testPredictionsAvg, axis=1)
        displayMetricsAgainstRandomGuessingMultiClass(np.argmax(yTrain, axis=1), np.argmax(yTest, axis=1), testPredictionsAvg, testClassPredictionsAvg, f"{prefix} LOGISTIC Mean Average", False)