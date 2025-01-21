import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from METRICS.METRIC_FOSSIL_CONFIDENCE import fossiliferousConfidenceScore
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# Define how many time slices we are looking at (aka temporal resolution)
# 50 = End of Triassic
# TODO: Investigate model results when changing slice count to reflect different time periods or eras
count = 50
# How many 'steps' we are looking at in the past for each temporal frame
step = 5
# Resolution of our data
# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 1
# How many features we are looking at (precipitation, temperature, elevation, koppen)
features = 4

# Define the path to resources given by our input
resPrefix = f"{1 / resolution}x{1 / resolution}"
# Retrieve the right resources based on the resolution
# TODO: REMOVE RESOURCE DEPENDENCIES FOR THESE CSV RESOURCE FILES, COMPILE NPY FILES INSTEAD
CLIMATE_PRECIPITATION_RESOURCE_DIR = pfl.RESOURCES_DIR / f"{resPrefix}MeanAnnualPrecipitation"
CLIMATE_TEMPERATURE_RESOURCE_DIR = pfl.RESOURCES_DIR / f"{resPrefix}MeanAnnualTemperatures"
DEM_RESOURCE_DIR = pfl.RESOURCES_DIR / f"{resPrefix}DEMs"
CLIMATE_KOPPEN_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}KoppenMaps"
PROJECTIONS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS"
GROUNDTRUTHS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}GROUNDTRUTHS"

# The 'name' of the fossil data we are looking at
targetData = "triassic_terrestrial_siliciclastic_occurences_resolutionScale_1_density_267"
# Extract the fossil labels aka temporal T1
fossilLabels = np.load(pfl.DATASET_DIR / "FOSSIL" / f"{targetData}.npy")

# count * 3 features + 1 label (for precipitation, temperature, DEM + fossil label)
climateDataset = np.zeros(((180 * resolution + 1) * (360 * resolution + 1), (count * features) + 1))
print(np.shape(climateDataset))
# Create a new dataset with the last column being the fossil labels
climateDataset[:, -1] = fossilLabels.flatten()

# Reverse loop through the time slices for chronological order (T50 -> T1) and adds each feature into the climateDataset
for i in tqdm(range(count - 1, -1, -1), desc="Adding Climate and Elevation Data"):

    # TODO: These need to be replaced with npy files

    # Read Precipitation data
    with open(os.path.join(CLIMATE_PRECIPITATION_RESOURCE_DIR,
                           pflh.getDirectoryFileNames(CLIMATE_PRECIPITATION_RESOURCE_DIR)[i]), "r") as f:
        dataPrecip = f.read()

    # Read Temperature data
    with open(os.path.join(CLIMATE_TEMPERATURE_RESOURCE_DIR,
                           pflh.getDirectoryFileNames(CLIMATE_TEMPERATURE_RESOURCE_DIR)[i]), "r") as f:
        dataTemp = f.read()

    # Read DEM (elevation) data
    with open(os.path.join(DEM_RESOURCE_DIR, pflh.getDirectoryFileNames(DEM_RESOURCE_DIR)[i]), "r") as f:
        dataDEM = f.read()

    # Load Koppen data and find the class for each cell
    koppenData = np.load(CLIMATE_KOPPEN_RESOURCE_DIR / pflh.getDirectoryFileNames(CLIMATE_KOPPEN_RESOURCE_DIR)[i])
    koppenClasses = np.argmax(koppenData, axis=-1)

    # Load the projected coordinates
    projectionData = np.load(
        PROJECTIONS_DATASET_DIR / f"{resPrefix}_projectionCoords_resolutionScale_{resolution}_T{i + 1}.npy")

    # Initialise arrays for precipitation, temperature, and elevation data
    precData = np.zeros((180 * resolution + 1) * (360 * resolution + 1))
    tempData = np.zeros((180 * resolution + 1) * (360 * resolution + 1))
    demData = np.zeros((180 * resolution + 1, 360 * resolution + 1))

    # Extract precipitation data
    rows = dataPrecip.split("\n")[1:-1]
    for j in range(len(rows)):
        precData[j] = float(rows[j].split(",")[2])

    # Extract temperature data
    rows = dataTemp.split("\n")[1:-1]
    for k in range(len(rows)):
        tempData[k] = float(rows[k].split(",")[2])

    # Extract elevation (DEM) data
    rows = dataDEM.split("\n")[:181]
    for latIdx, row in enumerate(rows):
        if row.strip():  # Ensure non-empty lines
            columns = row.split(",")[:-1]
            for longIdx, value in enumerate(columns):
                demData[latIdx, longIdx] = float(value)

    # Iterate over the projectionData, which holds every cell index
    for projectedIdx in range(len(projectionData)):
        # Get the index to look at
        dataIdx = int(projectionData[projectedIdx])

        # Get the relevant data by the projected index
        precipValue = precData[dataIdx]
        tempValue = tempData[dataIdx]
        demValue = demData.flatten()[dataIdx]
        koppenValue = koppenClasses.flatten()[dataIdx]

        #Assign precipitation, temperature, elevation, koppen data to the climate dataset
        climateDataset[projectedIdx, features * (count - 1 - i)] = precipValue
        climateDataset[projectedIdx, features * (count - 1 - i) + 1] = tempValue
        climateDataset[projectedIdx, features * (count - 1 - i) + 2] = demValue
        climateDataset[projectedIdx, features * (count - 1 - i) + 3] = koppenValue

# Define column names
columns = []
# We have 4 features (precipitation, temperature, elevation, koppen) = 1 kernel
# [0.2,0.3,0.5,0.2] [0]
# [0.3,0.4,0.6,0.4] [1]
# [0.4,0.5,0.7,0.9] [2]
for i in range(count - 1, -1, -1):
    columns.append(f"Precipitation_T{i + 1}")
    columns.append(f"Temperature_T{i + 1}")
    columns.append(f"Elevation_T{i + 1}")
    columns.append(f"Koppen_T{i + 1}")
columns.append("FossilLabel")

# View the dataframe to doublecheck we have populated columns
df = pd.DataFrame(climateDataset, columns=columns)
print(df.head())

# Load the ground truth map and create a mask using that and the fossil labels
groundTruth = np.load(GROUNDTRUTHS_DATASET_DIR / f"groundTruth_resolutionScale_{resolution}.npy").flatten()
# Create a mask to retain only true positives and true negatives
truePNMask = (fossilLabels.flatten() == 1) | (groundTruth == 0)

# Extract all the features and mask them using our ground truths to maximise training
cellFeatures = df.drop(columns=["FossilLabel"]).values[truePNMask]
cellLabels = fossilLabels.flatten()[truePNMask]

# Log the number of true positives and true negatives
print("True Positives (Fossils):", np.sum(cellLabels == 1))
print("True Negatives:", np.sum(cellLabels == 0))

# Normalise features
scaler = MinMaxScaler()
cellFeatures = scaler.fit_transform(cellFeatures)

# Reshape features into 3D (samples, timesteps, features)
cellFeatures = cellFeatures.reshape(cellFeatures.shape[0], count, features)

# Stratified split for train, test, and validation sets
xTrain, xVal, yTrain, yVal = train_test_split(cellFeatures, cellLabels, test_size=0.3, stratify=cellLabels, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, test_size=0.5, stratify=yVal, random_state=42)

# Check the lengths of the stratified dataset
print("Training set:", np.bincount(yTrain.astype(int)))
print("Validation set:", np.bincount(yVal.astype(int)))
print("Test set:", np.bincount(yTest.astype(int)))

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(yTrain),
    y=yTrain
)

# Create balanced weights
classWeights = {i: weights[i] for i in range(len(weights))}
print("Class Weights:", classWeights)

# Meat N Bones... The model
# A simple LSTM Model
tf.keras.backend.clear_session()
model =  keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(count, features)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(32, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train
history = model.fit(
    xTrain, yTrain,
    validation_data=(xVal, yVal),
    epochs=40,
    batch_size=32,
    class_weight=classWeights,
    callbacks=[callback],
    verbose=1
)

# TODO: REMOVE ALL THE BELOW CODE INTO A HELPER FUNCTION

# Evaluate the model on the test set
testPredictions = model.predict(xTest).flatten()
# Only consider predictions with a 90% or higher confidence
testBinaryPredictions = (testPredictions >= 0.9).astype(int)

# Calculate test metrics
print(f"Test Overall Accuracy: {accuracy_score(yTest, testBinaryPredictions):.4f}")
# How many of the predictions are actually true positives
print(f"Test Precision: {precision_score(yTest, testBinaryPredictions, zero_division=0):.4f}")
print(f"Test Recall: {recall_score(yTest, testBinaryPredictions, zero_division=0):.4f}")
print(f"Test Fossiliferous Confidence: {fossiliferousConfidenceScore(yTest, testPredictions):.4f}")

# Stratified random guessing
randomTestPredictions = np.random.choice([0, 1], len(yTest), True, [1 - np.mean(yTrain), np.mean(yTrain)])
print(f"Stratified Random Baseline Accuracy: {accuracy_score(yTest, randomTestPredictions):.4f}")
print(f"Stratified Random Baseline Precision: {precision_score(yTest, randomTestPredictions, zero_division=0):.4f}")
print(f"Stratified Random Baseline Fossiliferous Confidence: {fossiliferousConfidenceScore(yTest, randomTestPredictions):.4f}")
print(f"----------------------------")

# Normalise and reshape global features
globalCellFeatures = scaler.transform(climateDataset[:, :-1])
globalCellFeatures = globalCellFeatures.reshape(globalCellFeatures.shape[0], count, features)

# Predict globally
globalPredictions = model.predict(globalCellFeatures).flatten()
globalBinaryPredictions = (globalPredictions >= 0.9).astype(int)

# Calculate global metrics
print(f"Global Accuracy: {accuracy_score(fossilLabels, globalBinaryPredictions):.4f}")
print(f"Global Precision: {precision_score(fossilLabels, globalBinaryPredictions, zero_division=0):.4f}")
print(f"Global Recall: {recall_score(fossilLabels, globalBinaryPredictions, zero_division=0):.4f}")
print(f"Global Fossiliferous Confidence: {fossiliferousConfidenceScore(fossilLabels, globalPredictions):.4f}")
print(f"Global Random Guessing Baseline: {267 / 65431:.4f}")

globalPredictionsDF = pd.DataFrame({
    "Predicted Suitability": globalPredictions,
    "Actual Fossils Found": fossilLabels.flatten()
})

# Save the prediction with the recall as a unique identifier
globalPredictionsDF.to_csv(pfl.MODELS_OUTPUT_DIR / f"globalPredictions_LSTM_{recall_score(fossilLabels, globalBinaryPredictions, zero_division=0)}.csv", index=False)