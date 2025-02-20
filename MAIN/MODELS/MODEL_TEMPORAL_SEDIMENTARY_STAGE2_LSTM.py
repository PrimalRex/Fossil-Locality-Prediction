import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from METRICS.METRIC_FEATURE_IMPORTANCE import displayFeatureImportance
from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing
from MODEL_TEMPORAL_LOGISTIC_REGRESSIONS import logisticModel_T1, logisticModel_FlatVector, logisticModel_MeanAverage
from VISUALISERS.PLOT_TRAIN_HISTORY import plotAccuracy, plotLoss
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This model is the 2nd stage in which we do a fundamental approach to sedimentary vs non-sedimentary regional predictions using the dataset at hand
# This is then compared to a regression to investigate any temporal meaning in such a broad less-meaningful label set.

# Define how many time slices we are looking at (aka temporal resolution)
count = 50
# How many 'steps' we are looking at in the past for each temporal frame
step = 5
# Resolution of our data
# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 4
# How many features we are looking at (precipitation, temperature, elevation, koppen)
features = 11

# Define the path to resources given by our input
resPrefix = f"{1 / resolution}x{1 / resolution}"

# Retrieve the right resources based on the resolution
CLIMATE_PRECIPITATION_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}MeanAnnualPrecipitation"
CLIMATE_TEMPERATURE_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}MeanAnnualTemperatures"
DEM_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}DEMs"
FLOW_ACCUMULATION_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}FlowAccumulation"
WATER_FLUX_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}WaterFlux"
SEDIMENT_FLUX_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}SedimentFlux"
FLOOD_BASINS_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}FloodBasins"
EROSION_RATE_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}ErosionRate"
UPLIFT_RATE_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}UpliftRate"
SLOPE_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}Slope"
DEATHDENSITIES_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}DeathDensities"
PROJECTIONS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS"
FOSSIL_SEDIMENT_LABELS_DIR = pfl.DATASET_DIR / f"{resPrefix}SedimentaryRockCategories"


# Mask to ignore any ocean or ice cells and focus on terrestrial cells
oceanMask = np.load(FOSSIL_SEDIMENT_LABELS_DIR / f"OceanMaskBinary.npy", allow_pickle=True).flatten()
# Load the sediment labels, 0 = Ocean, 1 = Non-Sedimentary, 2 = Sedimentary
sedimentLabels = np.load(FOSSIL_SEDIMENT_LABELS_DIR / f"SedimentBinaryRegions.npy", allow_pickle=True).flatten()
# Convert values to binary labels
sedimentLabels[sedimentLabels == 1] = 0
sedimentLabels[sedimentLabels == 2] = 1

# count * X features + 1 label
# 'climate' refers to the entire dataset, not to be confused with climate variables
climateDataset = np.zeros(((180 * resolution + 1) * (360 * resolution + 1), (count * features) + 1))
climateDataset[:, -1] = sedimentLabels.flatten()

# Reverse loop through the time slices for chronological order (T50 -> T1) and adds each feature into the climateDataset
for i in tqdm(range(count - 1, -1, -1), desc="Compiling Dataset Features: "):

    # Load cellular datasets by their directory
    precData = np.load(CLIMATE_PRECIPITATION_RESOURCE_DIR / pflh.getDirectoryFileNames(CLIMATE_PRECIPITATION_RESOURCE_DIR)[i],
                       allow_pickle=True).flatten()
    demData = np.load(DEM_RESOURCE_DIR / pflh.getDirectoryFileNames(DEM_RESOURCE_DIR)[i],
                       allow_pickle=True).flatten()
    tempData = np.load(CLIMATE_TEMPERATURE_RESOURCE_DIR / pflh.getDirectoryFileNames(CLIMATE_TEMPERATURE_RESOURCE_DIR)[i],
                       allow_pickle=True).flatten()
    flowData = np.load(FLOW_ACCUMULATION_RESOURCE_DIR / pflh.getDirectoryFileNames(FLOW_ACCUMULATION_RESOURCE_DIR)[i],
                        allow_pickle=True).flatten()
    wfluxData = np.load(WATER_FLUX_RESOURCE_DIR / pflh.getDirectoryFileNames(WATER_FLUX_RESOURCE_DIR)[i],
                        allow_pickle=True).flatten()
    sfluxData = np.load(SEDIMENT_FLUX_RESOURCE_DIR / pflh.getDirectoryFileNames(SEDIMENT_FLUX_RESOURCE_DIR)[i],
                        allow_pickle=True).flatten()
    floodData = np.load(FLOOD_BASINS_RESOURCE_DIR / pflh.getDirectoryFileNames(FLOOD_BASINS_RESOURCE_DIR)[i],
                        allow_pickle=True).flatten()
    erosionData = np.load(EROSION_RATE_RESOURCE_DIR / pflh.getDirectoryFileNames(EROSION_RATE_RESOURCE_DIR)[i],
                        allow_pickle=True).flatten()
    upliftData = np.load(UPLIFT_RATE_RESOURCE_DIR / pflh.getDirectoryFileNames(UPLIFT_RATE_RESOURCE_DIR)[i],
                        allow_pickle=True).flatten()
    slopeData = np.load(SLOPE_RESOURCE_DIR / pflh.getDirectoryFileNames(SLOPE_RESOURCE_DIR)[i],
                        allow_pickle=True).flatten()
    deathData = np.load(DEATHDENSITIES_RESOURCE_DIR / pflh.getDirectoryFileNames(DEATHDENSITIES_RESOURCE_DIR)[i],
                        allow_pickle=True).flatten()

    # Load the projected coordinates
    projectionData = np.load(
        PROJECTIONS_DATASET_DIR / f"{resPrefix}_projectionCoords_resolutionScale_{resolution}_T{i + 1}.npy").flatten()

    offset = features * (count - 1 - i)
    for projectedIdx in range(len(projectionData)):
        # Get the index to look at
        dataIdx = int(projectionData[projectedIdx])
        precipValue = precData[dataIdx]
        demValue = demData[dataIdx]
        tempValue = tempData[dataIdx]
        flowValue = flowData[dataIdx]
        wfluxValue = wfluxData[dataIdx]
        sfluxValue = sfluxData[dataIdx]
        floodValue = floodData[dataIdx]
        erosionValue = erosionData[dataIdx]
        upliftValue = upliftData[dataIdx]
        slopeValue = slopeData[dataIdx]
        deathValue = deathData[dataIdx]

        # Assign values to the climate dataset
        climateDataset[projectedIdx, offset] = precipValue
        climateDataset[projectedIdx, offset + 1] = demValue
        climateDataset[projectedIdx, offset + 2] = tempValue
        climateDataset[projectedIdx, offset + 3] = flowValue
        climateDataset[projectedIdx, offset + 4] = wfluxValue
        climateDataset[projectedIdx, offset + 5] = sfluxValue
        climateDataset[projectedIdx, offset + 6] = floodValue
        climateDataset[projectedIdx, offset + 7] = erosionValue
        climateDataset[projectedIdx, offset + 8] = upliftValue
        climateDataset[projectedIdx, offset + 9] = slopeValue
        climateDataset[projectedIdx, offset + 10] = deathValue

# Define column names
columns = []
# We have X features (precipitation, temperature, elevation... ) = 1 kernel
# [0.2,0.3,0.5,0.2] [0]
# [0.3,0.4,0.6,0.4] [1]
# [0.4,0.5,0.7,0.9] [2]
for i in range(count - 1, -1, -1):
    columns.append(f"Precipitation_T{i + 1}")
    columns.append(f"Elevation_T{i + 1}")
    columns.append(f"Temperature_T{i + 1}")
    columns.append(f"FlowAccumulation_T{i + 1}")
    columns.append(f"WaterFlux_T{i + 1}")
    columns.append(f"SedimentFlux_T{i + 1}")
    columns.append(f"FloodBasins_T{i + 1}")
    columns.append(f"ErosionRate_T{i + 1}")
    columns.append(f"UpliftRate_T{i + 1}")
    columns.append(f"Slope_T{i + 1}")
    columns.append(f"DeathDensities_T{i + 1}")
columns.append("SedimentaryLabel")

# Filter the dataset by oceanMask
filteredClimateDataset = climateDataset[oceanMask == 1]

# View the dataframe to doublecheck we have populated columns
df = pd.DataFrame(filteredClimateDataset, columns=columns)
print(df.head())
print("DataFrame size:", df.shape)

cellLabels = df["SedimentaryLabel"].values.flatten()
cellFeatures = df.drop("SedimentaryLabel", axis=1).values

# Log the number of true positives and true negatives
print("True Positives: ", np.sum(cellLabels == 1))
print("True Negatives: ", np.sum(cellLabels == 0))

# Find the feature importance of the dataset
# displayFeatureImportance(cellFeatures, cellLabels, df)

# Stratified split for train/test/val sets
xTrain, xVal, yTrain, yVal = train_test_split(cellFeatures, cellLabels, test_size=0.3, stratify=cellLabels, shuffle=True, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, test_size=0.3, stratify=yVal, shuffle=True, random_state=42)

# Compute class weights
classWeights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(yTrain),
    y=yTrain
)
# Convert to dictionary
classWeights = dict(enumerate(classWeights))

# Initialize the scaler
scaler = StandardScaler()
# Flatten to features and then reshape to the regular format for RNN
xTrain = scaler.fit_transform(xTrain.reshape(-1, features)).reshape(-1, count, features)
xVal = scaler.transform(xVal.reshape(-1, features)).reshape(-1, count, features)
xTest = scaler.transform(xTest.reshape(-1, features)).reshape(-1, count, features)

# Check the lengths of the sets
print("Training set:", np.bincount(yTrain.astype(int)))
print("Validation set:", np.bincount(yVal.astype(int)))
print("Test set:", np.bincount(yTest.astype(int)))

# Main Model
tf.keras.backend.clear_session()
model =  keras.models.Sequential()
model.add(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(count, features)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.05))

# Second LSTM Layer
model.add(tf.keras.layers.LSTM(160, return_sequences=False))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))

# Dense Layers for Final Classification
model.add(tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

# Check the model summary
model.summary()

# Train
history = model.fit(
    xTrain, yTrain,
    validation_data=(xVal, yVal),
    epochs=20,
    batch_size=256,
    class_weight=classWeights,
    callbacks=[earlyStopping, lrScheduler],
    verbose=1
)

# Plot training graphs
plotAccuracy(history)
plotLoss(history)

# # Can load weights into the model to test a model (Essentially loading a pretrained version of the model)
# #model.load_weights(pfl.MODELS_OUTPUT_DIR / "globalPredictions_fossiliferousSedimentary_LSTM_0.9611.h5")

# Evaluate the model on the test set
testPredictions = model.predict(xTest).flatten()
testBinaryPredictions = (testPredictions >= 0.90).astype(int)

displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictions, testBinaryPredictions, "Sedimentary Vs Non-Sedimentary LSTM")

# testPredictionsDF = pd.DataFrame({
#     "Predicted Sedimentary Suitability": testPredictions,
#     "Actual Sedimentary Labels": yTest.flatten()
# })
#
# # Save the prediction with the recall as a unique identifier
# outName = f"testPredictions_binarySedimentary_LSTM_{accuracy_score(yTest, testBinaryPredictions):.4f}"
# # outputPath = pathlib.Path(pfl.MODELS_OUTPUT_DIR) / f"{outName}.csv"
# # testPredictionsDF.to_csv(outputPath, index=False)
#
# # Save the weights to the same outpath
# outputPath = pathlib.Path(pfl.MODELS_OUTPUT_DIR) / f"{outName}.h5"
# model.save_weights(outputPath)

# # Compare metrics against the logistic models
# logisticModel_T1(xTrain, yTrain, xTest, yTest, 0.90,"Sedimentary Vs Non-Sedimentary")
# logisticModel_FlatVector(xTrain, yTrain, xTest, yTest, 0.90,"Sedimentary Vs Non-Sedimentary")
# logisticModel_MeanAverage(xTrain, yTrain, xTest, yTest, 0.90, "Sedimentary Vs Non-Sedimentary")