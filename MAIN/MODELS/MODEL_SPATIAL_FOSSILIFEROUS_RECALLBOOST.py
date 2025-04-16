import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from METRICS.METRIC_FEATURE_IMPORTANCE import displayFeatureImportanceSingleTimestep
from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing, getMetrics
from METRICS.METRIC_IOU import IoUScore
from MODEL_TEMPORAL_LOGISTIC_REGRESSIONS import logisticModel_T1, logisticModel_FlatVector, logisticModel_MeanAverage
from VISUALISERS.PLOT_SPATIAL_THRESHOLD import displayThresholdImpact
from VISUALISERS.PLOT_TRAIN_HISTORY import plotAccuracy, plotLoss
from MODEL_KERASTUNER import BayesianBinaryOptim, HyperbandBinaryOptim
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# FUNCTIONS -------------------------------------------------------------------

# Function to subdivide an even map to produce smaller trainable elements
def subdivideMap(featuresMap, labelMap, size, featuresCount):
    height, width = featuresMap.shape[:2]

    # Calculate number of cubes that can be made
    cubeCount = height // size * width // size

    # Create new arrays to hold the subdivided data
    subdivFeatures = np.zeros((cubeCount, size, size, featuresCount))
    subdivLabel = np.zeros((cubeCount, size, size))

    # Extract cubes
    idx = 0
    for i in range(0, height, size):
        for j in range(0, width, size):
            # Extract cube from features map
            subdivFeatures[idx] = featuresMap[i:i + size, j:j + size, :]

            # Extract corresponding label cube
            subdivLabel[idx] = labelMap[i:i + size, j:j + size]

            # Increment
            idx += 1

    return subdivFeatures, subdivLabel

# Function to linear upscale predictions to full resolution to be used in conjunction with the temporal ouput
def reconstructCubedMap(predictions, targetHeight, targetWidth, size):
    # Create a new array to hold the final map
    finalMap = np.zeros((targetHeight, targetWidth))

    # Calculate number of cubes horizontally
    cubeInRow = targetWidth // size

    for cubeIdx in range(len(predictions)):
        # Start from the top left
        rowIdx = (cubeIdx // cubeInRow) * size
        colIdx = (cubeIdx % cubeInRow) * size

        # Assign the same value across the entire sizeXsize block
        finalMap[rowIdx:rowIdx + size, colIdx:colIdx + size] = predictions[cubeIdx]

    return finalMap

# MAIN ------------------------------------------------------------------------

# This model behaves as a recall booster, we do a simple contemporaneous prediction on the full occurrence label set to find
# recognisable exploration patterns which in turn can boost our temporal Stage 4 model performance.

# Resolution of our data
# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 4
# How many features we are looking at (precipitation, temperature, elevation, koppen)
features = 9
# Which period we should be looking to tune towards
geologicPrefix = "ALLPERIODS"

# Define the path to resources given by our input
resPrefix = f"{1 / resolution}x{1 / resolution}"

# HYPERPARAMETER OPTIMISATION -------------------------------------------------------------------

# Model to be passed into keras fine tuner
def buildSkeleton(hp):
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(cubeSize, cubeSize, features)))

    # First Conv Block
    model.add(tf.keras.layers.Conv2D(192, (3, 3), activation="relu", padding="same",
                                     kernel_regularizer=tf.keras.regularizers.l2(hp.Float("kernel_regularizer_1", min_value=1e-5, max_value=1e-2, sampling="log"))))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(hp.Float("dropout_1", min_value=0.05, max_value=0.5, step=0.05)))

    # Second Conv Block
    model.add(tf.keras.layers.Conv2D(384, (3, 3), activation="relu", padding="same",
                                     kernel_regularizer=tf.keras.regularizers.l2(hp.Float("kernel_regularizer_2", min_value=1e-5, max_value=1e-2, sampling="log"))))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(hp.Float("dropout_2", min_value=0.05, max_value=0.5, step=0.05)))

    # Third Conv Block
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same",
                                     kernel_regularizer=tf.keras.regularizers.l2(hp.Float("kernel_regularizer_3", min_value=1e-5, max_value=1e-2, sampling="log"))))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(hp.Float("dropout_3", min_value=0.05, max_value=0.5, step=0.05)))

    # Final Dense Layers for Classification
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(
        hp.Float("kernel_regularizer_4", min_value=1e-5, max_value=1e-2, sampling="log"))))
    model.add(tf.keras.layers.Dropout(hp.Float("dropout_4", min_value=0.05, max_value=0.5, step=0.05)))

    model.add(tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(
        hp.Float("kernel_regularizer_5", min_value=1e-5, max_value=1e-2, sampling="log"))))
    model.add(tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(
        hp.Float("kernel_regularizer_6", min_value=1e-5, max_value=1e-2, sampling="log"))))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log"))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# MAIN ------------------------------------------------------------------------

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
NDVI_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}NDVI"
PROJECTIONS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS"
FOSSIL_SEDIMENT_LABELS_DIR = pfl.DATASET_DIR / f"{resPrefix}SedimentaryRockCategories"
FOSSIL_TRUTH_LABELS_DIR = pfl.DATASET_DIR / f"{resPrefix}FossilOccurrences"
MODEL_PREDICTIONS_DIR = pfl.PREDICTIONS_DIR / f"{resPrefix}SpatialModel"
pflh.createDirectoryIfNotExist(MODEL_PREDICTIONS_DIR)

# Define column names (features names)
columns = []
columns.append(f"Precipitation")
columns.append(f"Elevation")
columns.append(f"Temperature")
columns.append(f"FlowAccumulation")
columns.append(f"WaterFlux")
columns.append(f"SedimentFlux")
#columns.append(f"FloodBasins")
columns.append(f"ErosionRate")
columns.append(f"UpliftRate")
#columns.append(f"Slope")
columns.append(f"NDVI")
columns.append("FossiliferousLabel")


# Load the fossiliferous labels
geologicFocusedLabels = np.load(FOSSIL_TRUTH_LABELS_DIR / f"{geologicPrefix.upper()}_FULL_OCCURRENCES.npy", allow_pickle=True).flatten()

# count * X features + 1 label
# 'climate' refers to the entire dataset, not to be confused with climate variables
climateDataset = np.zeros(((180 * resolution + 1) * (360 * resolution + 1), features + 1))
climateDataset[:, -1] = geologicFocusedLabels.flatten()

# Load the closest timestep in all feature directories
precData = np.load(CLIMATE_PRECIPITATION_RESOURCE_DIR / pflh.getDirectoryFileNames(CLIMATE_PRECIPITATION_RESOURCE_DIR)[0],
                   allow_pickle=True).flatten()
demData = np.load(DEM_RESOURCE_DIR / pflh.getDirectoryFileNames(DEM_RESOURCE_DIR)[0],
                   allow_pickle=True).flatten()
tempData = np.load(CLIMATE_TEMPERATURE_RESOURCE_DIR / pflh.getDirectoryFileNames(CLIMATE_TEMPERATURE_RESOURCE_DIR)[0],
                   allow_pickle=True).flatten()
flowData = np.load(FLOW_ACCUMULATION_RESOURCE_DIR / pflh.getDirectoryFileNames(FLOW_ACCUMULATION_RESOURCE_DIR)[0],
                    allow_pickle=True).flatten()
wfluxData = np.load(WATER_FLUX_RESOURCE_DIR / pflh.getDirectoryFileNames(WATER_FLUX_RESOURCE_DIR)[0],
                    allow_pickle=True).flatten()
sfluxData = np.load(SEDIMENT_FLUX_RESOURCE_DIR / pflh.getDirectoryFileNames(SEDIMENT_FLUX_RESOURCE_DIR)[0],
                    allow_pickle=True).flatten()
floodData = np.load(FLOOD_BASINS_RESOURCE_DIR / pflh.getDirectoryFileNames(FLOOD_BASINS_RESOURCE_DIR)[0],
                    allow_pickle=True).flatten()
erosionData = np.load(EROSION_RATE_RESOURCE_DIR / pflh.getDirectoryFileNames(EROSION_RATE_RESOURCE_DIR)[0],
                    allow_pickle=True).flatten()
upliftData = np.load(UPLIFT_RATE_RESOURCE_DIR / pflh.getDirectoryFileNames(UPLIFT_RATE_RESOURCE_DIR)[0],
                    allow_pickle=True).flatten()
slopeData = np.load(SLOPE_RESOURCE_DIR / pflh.getDirectoryFileNames(SLOPE_RESOURCE_DIR)[0],
                    allow_pickle=True).flatten()
ndviData = np.load(NDVI_RESOURCE_DIR / pflh.getDirectoryFileNames(NDVI_RESOURCE_DIR)[0],
                    allow_pickle=True).flatten()

# Append all to the climate dataset
climateDataset[:, 0] = precData
climateDataset[:, 1] = demData
climateDataset[:, 2] = tempData
climateDataset[:, 3] = flowData
climateDataset[:, 4] = wfluxData
climateDataset[:, 5] = sfluxData
#climateDataset[:, 6] = floodData
climateDataset[:, 6] = erosionData
climateDataset[:, 7] = upliftData
#climateDataset[:, 9] = slopeData
climateDataset[:, 8] = ndviData

# View the dataframe to doublecheck we have populated columns
df = pd.DataFrame(climateDataset, columns=columns)
print(df.head())
print("DataFrame size:", df.shape)

# Load the dataframe from the .npy file
print(df.head())
cellLabels = df["FossiliferousLabel"].values.flatten()
cellFeatures = df.drop("FossiliferousLabel", axis=1).values

# Find the feature importance of the dataset
# displayFeatureImportanceSingleTimestep(cellFeatures, cellLabels, df, "FossiliferousLabel")

# Log the number of true positives and true negatives
print("True Positives: ", np.sum(cellLabels == 1))
print("True Negatives: ", np.sum(cellLabels == 0))

# Iterate through all the features and purge the first column and first row
reshapedFeatures = []
for i in range(features):
    # Reshape to map form
    featureMap = cellFeatures[:, i].reshape(721, 1441)
    # Drop first row and first column to make a safely divisible resolution
    featureMap = featureMap[1:, 1:]
    reshapedFeatures.append(featureMap.flatten())

# Stack the reshaped features
cellFeatures = np.column_stack(reshapedFeatures)

# Purge first column and row for labels too
labelsMap = cellLabels.reshape(721, 1441)
labelsMap = labelsMap[1:, 1:]
cellLabels = labelsMap.flatten()

# Scale each feature column (potentially leaky)
scaler = StandardScaler()
cellFeatures = scaler.fit_transform(cellFeatures)

# Reshape back into map form
mapFeatures = cellFeatures.reshape((180 * resolution), (360 * resolution), features)
mapLabels = cellLabels.reshape((180 * resolution), (360 * resolution))

# Subdivide the features and labels into 10x10 resolution grids
cubeSize = 10
subdivMapFeatures, subdivMapLabels = subdivideMap(mapFeatures, mapLabels, cubeSize, features)

# Create the aggregate binary labels to create per subdiv true and negative
aggregateLabels = (np.mean(subdivMapLabels, axis=(1,2)) > 0.1).astype(int)
print(f"Positive cubes: {np.sum(aggregateLabels == 1)}")
print(f"Negative cubes: {np.sum(aggregateLabels == 0)}")

# Split into train/val/test as normal
xTrain, xVal, yTrain, yVal = train_test_split(subdivMapFeatures, aggregateLabels, test_size=0.3, stratify=aggregateLabels, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, test_size=0.6, stratify=yVal, random_state=42)

# (OPTIONAL) Full training split train/test sets (No validation set)
xTrain = np.concatenate((xTrain, xVal))
yTrain = np.concatenate((yTrain, yVal))

# Check the lengths of the sets
print("Training set:", np.bincount(yTrain.astype(int)))
print("Validation set:", np.bincount(yVal.astype(int)))
print("Test set:", np.bincount(yTest.astype(int)))

classWeights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(yTrain),
    y=yTrain
)
# Convert to dictionary
classWeights = dict(enumerate(classWeights))

# (OPTIONAL) Hyperparameter optimisation
# HyperbandBinaryOptim(buildSkeleton, xTrain, yTrain, xVal, yVal, xTest, yTest, maxEpochs=100, binaryThreshold=0.90, batchSize=32, prefix="RECALL_BOOST")

# # Main Model
# tf.keras.backend.clear_session()
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(cubeSize, cubeSize, features)))
#
# # First Conv Block
# model.add(tf.keras.layers.Conv2D(192, (3, 3), activation="relu", padding="same",
#                                  kernel_regularizer=tf.keras.regularizers.l2(0.0027067)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Dropout(0.25))
#
# # Second Conv Block
# model.add(tf.keras.layers.Conv2D(384, (3, 3), activation="relu", padding="same",
#                                  kernel_regularizer=tf.keras.regularizers.l2(0.0020273)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Dropout(0.1))
#
# # Third Conv Block
# model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same",
#                                  kernel_regularizer=tf.keras.regularizers.l2(8.7848e-05)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Dropout(0.2))
#
# # Final Dense Layers for Classification
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.00010293)))
# model.add(tf.keras.layers.Dropout(0.1))
# model.add(tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(2.4261e-05)))
# model.add(tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1.3679e-05)))
# model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=4.3554e-05, clipnorm=1.0)
# model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
# earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
#
# # Check the model summary
# model.summary()
#
# # Train
# history = model.fit(
#     xTrain, yTrain,
#     #validation_data=(xVal, yVal),
#     epochs=60,
#     batch_size=32,
#     class_weight=classWeights,
#     #callbacks=[earlyStopping],
#     verbose=1)
#
# # Plot training graphs
# #plotAccuracy(history)
# #plotLoss(history)
#
# # # Predict and reshape into cubes
# # preds = model.predict(subdivMapFeatures).flatten()
# # binary_preds = (preds >= 0.3).astype(int)
# #
# # full_map = reconstruct_full_resolution_map(binary_preds)
# # # Plot the reconstructed map
# # plt.figure(figsize=(10, 5))
# # plt.imshow(full_map, cmap='viridis', interpolation='nearest')
# # plt.grid(False)
# # plt.tight_layout()
# # plt.show()
#
# testPredictions = model.predict(xTest).flatten()
# testBinaryPredictions = (testPredictions >= 0.30).astype(int)
# displayMetricsAgainstRandomGuessing(yTest, yTest, testPredictions, testBinaryPredictions, "Recall Boost CNN", True)



# Use Stratified KFold to produce the Final Recall Map
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Create the array that will hold all predictions
globalPredictions = np.zeros(len(aggregateLabels))
globalPredictionsRaw = np.zeros(len(aggregateLabels))

# Perform cross-validation
for fold, (trainIdx, valIdx) in enumerate(skf.split(subdivMapFeatures, aggregateLabels)):
    print(f"\nComputing Fold: {fold + 1} / {5}")

    # Split data into val and train
    xTrain, xVal = subdivMapFeatures[trainIdx], subdivMapFeatures[valIdx]
    yTrain, yVal = aggregateLabels[trainIdx], aggregateLabels[valIdx]

    # Compute class weights
    classWeights = compute_class_weight(class_weight="balanced",classes=np.unique(yTrain),y=yTrain)
    classWeights = dict(enumerate(classWeights))

    # Main Model condensed
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(cubeSize, cubeSize, features)))
    model.add(tf.keras.layers.Conv2D(192, (3, 3), activation="relu", padding="same",
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0027067)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(384, (3, 3), activation="relu", padding="same",
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0020273)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same",
                                     kernel_regularizer=tf.keras.regularizers.l2(8.7848e-05)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.00010293)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(2.4261e-05)))
    model.add(tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1.3679e-05)))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=4.3554e-05, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Train
    model.fit(xTrain, yTrain, epochs=60, batch_size=32, class_weight=classWeights, verbose=1)

    # Predict on the validation set
    valPredictions = model.predict(xVal).flatten()
    valBinaryPredictions = (valPredictions >= 0.50).astype(int)

    # Store the predictions in their correct index locations
    globalPredictions[valIdx] = valBinaryPredictions
    globalPredictionsRaw[valIdx] = valPredictions

# Perform post-cv independent evaluation
print("IoU Score: ", IoUScore(aggregateLabels.flatten(), globalPredictions.flatten()))
# Display the threshold impact
displayThresholdImpact(aggregateLabels, globalPredictionsRaw, 0.1, 0.9, 0.05)

# Reconstruct the full map from the predictions
predictedMap = reconstructCubedMap(globalPredictions, 180 * resolution, 360 * resolution, cubeSize)
# Create the padded map
paddedMap = np.zeros((180 * resolution + 1, 360 * resolution + 1))
# Assign the predicted map to the padded map
paddedMap[1:, 1:] = predictedMap

# Export the padded map
np.save(MODEL_PREDICTIONS_DIR / f"{geologicPrefix.upper()}_RECALL_BOOST_V2.npy", paddedMap)

# Plot the reconstructed map
plt.figure(figsize=(10, 5))
plt.imshow(paddedMap, cmap="viridis", interpolation="nearest")
plt.grid(False)
plt.tight_layout()
plt.show()