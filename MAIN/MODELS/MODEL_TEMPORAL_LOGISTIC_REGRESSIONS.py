import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This is a script to train logistic models for the temporal aspect of the predictions
# Models assume that the data splits have already been scaled appropriately before invocation

# This model trains on a single-timestep, usually the T1 timestep to appreciate whether temporal data is relevant for the classification problem
def logisticModel_T1(xTrain, yTrain, xTest, yTest, binaryThreshold=0.90, prefix="Test"):
    # Extract last timestep (T1)
    xTrainT1 = xTrain[:, -1, :]
    xTestT1 = xTest[:, -1, :]

    logReg = LogisticRegression(max_iter=1000)
    logReg.fit(xTrainT1, yTrain)

    testPredictions = logReg.predict_proba(xTestT1)[:, 1]
    testBinaryPredictions = (testPredictions >= binaryThreshold).astype(int)
    displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictions, testBinaryPredictions, f"{prefix} LOGISTIC T1")

# This model flattens all the timesteps into a single feature vector to appreciate temporal order, and if temporal importance is relevant for the classification problem
def logisticModel_FlatVector(xTrain, yTrain, xTest, yTest, binaryThreshold=0.90, prefix="Test"):
    xTrainFlatVector = xTrain.reshape(xTrain.shape[0], -1)
    xTestFlatVector = xTest.reshape(xTest.shape[0], -1)

    logRegFlatVector = LogisticRegression(max_iter=1000)
    logRegFlatVector.fit(xTrainFlatVector, yTrain)

    testPredictionsFlatVector = logRegFlatVector.predict_proba(xTestFlatVector)[:, 1]
    testBinaryPredictionsFlatVector = (testPredictionsFlatVector >= binaryThreshold).astype(int)
    displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictionsFlatVector, testBinaryPredictionsFlatVector, f"{prefix} LOGISTIC Flat Vector")


# This model averages all the timesteps into a single timestep T(X) to appreciate if long-term data varies in comparison to short-term data
def logisticModel_MeanAverage(xTrain, yTrain, xTest, yTest, binaryThreshold=0.90, prefix="Test"):
    xTrainAvg = np.mean(xTrain, axis=1)
    xTestAvg = np.mean(xTest, axis=1)

    logRegAvg = LogisticRegression(max_iter=1000)
    logRegAvg.fit(xTrainAvg, yTrain)

    testPredictionsAvg = logRegAvg.predict_proba(xTestAvg)[:, 1]
    testBinaryPredictionsAvg = (testPredictionsAvg >= binaryThreshold).astype(int)
    displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictionsAvg, testBinaryPredictionsAvg,f"{prefix} LOGISTIC Mean Average")