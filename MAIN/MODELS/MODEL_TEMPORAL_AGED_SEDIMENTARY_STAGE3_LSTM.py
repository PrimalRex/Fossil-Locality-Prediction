import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from METRICS.METRIC_FEATURE_IMPORTANCE import displayFeatureImportance
from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing, displayMetricsAgainstRandomGuessingMultiClass
from MODEL_TEMPORAL_LOGISTIC_REGRESSIONS import logisticModel_T1, logisticModel_FlatVector, logisticModel_MeanAverage
from VISUALISERS.PLOT_TRAIN_HISTORY import plotAccuracy, plotLoss
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# FUNCTIONS -------------------------------------------------------------------

# Model to be passed into keras fine tuner
#def buildSkeleton(hp):

# Categorises the age of the sedimentary rock into an applicable range within the 5 bands with +- 2MYr buffer
def categoriseAge(value):
    if value >= 250:
        return 5  # Pre-Mesozoic
    elif value >= 199:
        return 4  # Triassic
    elif value >= 141:
        return 3  # Jurassic
    elif value >= 64:
        return 2  # Cretaceous
    elif value >= 21:
        return 1  # Cenozoic
    else:
        return 0  # None (Pre-Mesozoic or Post-Cenozoic or N/A)

# MAIN ------------------------------------------------------------------------

# This is the 3rd stage in which we do further analysis into identifying geological sedimentary regions.
# These regions hone into the acceptable bounding area's for time-specific occurrences.
# This, like the previous stage, compares to logistic regression models to appreciate the temporal aspect of the data.

# Define how many time slices we are looking at (aka temporal resolution)
count = 50
# How many 'steps' we are looking at in the past for each temporal frame
step = 5
# Resolution of our data
# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 4
# How many features we are looking at (precipitation, temperature, elevation, koppen)
features = 10
# If we want to do a fresh dataframe construction or use a cached version
loadCompiledDataframe = True

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
PROJECTIONS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS"
FOSSIL_SEDIMENT_LABELS_DIR = pfl.DATASET_DIR / f"{resPrefix}SedimentaryRockCategories"
COMPILED_DATAFRAMES_DIR = pfl.DATASET_DIR / f"CompiledDataFrames"
pflh.createDirectoryIfNotExist(COMPILED_DATAFRAMES_DIR)

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
columns.append("AgedSedimentaryLabel")

if not loadCompiledDataframe:
    # Mask to ignore any ocean or ice cells and focus on terrestrial cells
    oceanMask = np.load(FOSSIL_SEDIMENT_LABELS_DIR / f"OceanMaskBinary.npy", allow_pickle=True).flatten()
    # Load the aged sedimentary labels
    agedSedimentaryLabels = np.load(FOSSIL_SEDIMENT_LABELS_DIR / f"approxMyaSedimentRegions.npy", allow_pickle=True).flatten()
    #print(np.shape(agedSedimentaryLabels))

    # Categorise the ages into 5 integer categories (ready for one-hot)
    agedSedimentaryLabels = np.array([categoriseAge(age) for age in agedSedimentaryLabels])

    # count * X features + 1 label
    # 'climate' refers to the entire dataset, not to be confused with climate variables
    climateDataset = np.zeros(((180 * resolution + 1) * (360 * resolution + 1), (count * features) + 1))
    climateDataset[:, -1] = agedSedimentaryLabels.flatten()

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

    # Filter the dataset by oceanMask
    filteredClimateDataset = climateDataset[oceanMask == 1]

    # View the dataframe to doublecheck we have populated columns
    df = pd.DataFrame(filteredClimateDataset, columns=columns)
    print(df.head())
    print("DataFrame size:", df.shape)

    # Save the dataframe to a .npy file for caching
    print("Saving Dataframe to disk...")
    np.save(COMPILED_DATAFRAMES_DIR / f"{resPrefix}CategoricalAgedSedimentary(STAGE3)Dataset.npy", df.values)

# Load the dataframe from the .npy file
df = pd.DataFrame(np.load(COMPILED_DATAFRAMES_DIR / f"{resPrefix}CategoricalAgedSedimentary(STAGE3)Dataset.npy"), columns=columns)
print(df.head())
cellLabels = df["AgedSedimentaryLabel"].values.flatten()
# Convert to one-hot encoding
cellLabelsOneHot = to_categorical(cellLabels, num_classes=np.unique(cellLabels).shape[0])
cellFeatures = df.drop("AgedSedimentaryLabel", axis=1).values

# Find the feature importance of the dataset
# displayFeatureImportance(cellFeatures, np.argmax(cellLabelsOneHot, axis=1), df, "AgedSedimentaryLabel")

# Stratified split for train/test/val sets
xTrain, xVal, yTrain, yVal = train_test_split(cellFeatures, cellLabelsOneHot, test_size=0.3, stratify=cellLabels, shuffle=True, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, test_size=0.6, stratify=np.argmax(yVal, axis=1), shuffle=True, random_state=42)

# (OPTIONAL) Full training split train/test sets (No validation set)
# xTrain = np.concatenate((xTrain, xVal))
# yTrain = np.concatenate((yTrain, yVal))

# Compute class weights
classWeights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(np.argmax(yTrain, axis=1)),
    y=np.argmax(yTrain, axis=1)
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
print("Training set:", np.bincount(np.argmax(yTrain, axis=1)))
print("Validation set:", np.bincount(np.argmax(yVal, axis=1)))
print("Test set:", np.bincount(np.argmax(yTest, axis=1)))

# (OPTIONAL) Run preliminary logistic regression models to see if there's any temporal importance in the dataset
#logisticModel_T1(xTrain, yTrain, xTest, yTest, prefix="Aged Sedimentary", multiClass=True)
#logisticModel_FlatVector(xTrain, yTrain, xTest, yTest, prefix="Aged Sedimentary", multiClass=True)
#logisticModel_MeanAverage(xTrain, yTrain, xTest, yTest, prefix="Aged Sedimentary", multiClass=True)

# Main Model
tf.keras.backend.clear_session()
model = keras.models.Sequential()
model.add(tf.keras.layers.LSTM(480, return_sequences=True, input_shape=(count, features)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

# Second LSTM Layer
model.add(tf.keras.layers.LSTM(256, return_sequences=False))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.1))

# Dense Layers for Final Classification
model.add(tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(6, activation="softmax"))

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00013562501444661616, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

# Check the model summary
model.summary()

# Train
# history = model.fit(
#     xTrain, yTrain,
#     validation_data=(xVal, yVal),
#     epochs=50,
#     batch_size=256,
#     class_weight=classWeights,
#     callbacks=[earlyStopping, lrScheduler],
#     verbose=1
# )

# Plot training graphs
#plotAccuracy(history)
#plotLoss(history)

# Can load weights into the model to test a model (Essentially loading a pretrained version of the model)
model.load_weights(pfl.MODELS_OUTPUT_DIR / "testPredictions_categoricalAgedSedimentary_LSTM_0.8463.h5")

# Evaluate the model on the test set
testPredictions = model.predict(xTest)
testClassPredictions = np.argmax(testPredictions, axis=1)
displayMetricsAgainstRandomGuessingMultiClass(np.argmax(yTrain, axis=1), np.argmax(yTest, axis=1), testPredictions, testClassPredictions, "Aged Sedimentary")

# Save the weights to the output directory
outName = f"testPredictions_categoricalAgedSedimentary_LSTM_{accuracy_score(np.argmax(yTest, axis=1), testClassPredictions):.4f}"
outputPath = pathlib.Path(pfl.MODELS_OUTPUT_DIR) / f"{outName}.h5"
#model.save_weights(outputPath)

# Get classification report to inspect each category
classes = ["N/A", "Ceno.", "Cret.", "Juras.", "Trias.", "Pre-Meso."]
print(classification_report(np.argmax(yTest, axis=1), testClassPredictions, target_names=classes, digits=4))
cm = confusion_matrix( np.argmax(yTest, axis=1), testClassPredictions)
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
plt.figure(figsize=(15, 15))
cmDisplay.plot()
plt.title("Confusion Matrix for Aged Sedimentary Classification")
plt.show()
