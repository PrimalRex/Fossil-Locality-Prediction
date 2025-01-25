import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from METRICS.METRIC_FOSSIL_CONFIDENCE import fossiliferousConfidenceScore
from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This model primarily focuses on the temporal learning of sedimentary data, this LSTM model is used to understand any
# and all environmental factors that may have influenced the sedimentary bedrock as our foundation for fossil plausibility.

# Define how many time slices we are looking at (aka temporal resolution)
count = 50
# How many 'steps' we are looking at in the past for each temporal frame
step = 5
# Resolution of our data
# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 4
# How many features we are looking at (precipitation, temperature, elevation, koppen)
features = 2

# Define the path to resources given by our input
resPrefix = f"{1 / resolution}x{1 / resolution}"

# Retrieve the right resources based on the resolution
CLIMATE_PRECIPITATION_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}MeanAnnualPrecipitation"
CLIMATE_TEMPERATURE_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}MeanAnnualTemperatures"
DEM_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}DEMs"
#CLIMATE_KOPPEN_RESOURCE_DIR = pfl.DATASET_DIR / f"{resPrefix}KoppenMaps"
PROJECTIONS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS"

# Create our label set
sedimentaryLabels = Image.open(os.path.join((pfl.RESOURCES_DIR / f"1.0x1.0SuitabilityMasks"), (f"{resPrefix}SedimentaryGeology.PNG"))).convert("L")
sedimentaryLabels = np.array(sedimentaryLabels)
sedimentaryLabels = (sedimentaryLabels > 128).astype(int)

# count * 3 features + 1 label (for precipitation, temperature, DEM + fossil label)
climateDataset = np.zeros(((180 * resolution + 1) * (360 * resolution + 1), (count * features) + 1))
print(np.shape(climateDataset))
climateDataset[:, -1] = sedimentaryLabels.flatten()

# Reverse loop through the time slices for chronological order (T50 -> T1) and adds each feature into the climateDataset
for i in tqdm(range(count - 1, -1, -1), desc="Adding Climate and Elevation Data"):

    # TODO: These need to be replaced with npy files

    # Load Koppen data and find the class for each cell
    # koppenData = np.load(CLIMATE_KOPPEN_RESOURCE_DIR / pflh.getDirectoryFileNames(CLIMATE_KOPPEN_RESOURCE_DIR)[i])
    # koppenClasses = np.argmax(koppenData, axis=-1)

    precData = np.load(CLIMATE_PRECIPITATION_RESOURCE_DIR / pflh.getDirectoryFileNames(CLIMATE_PRECIPITATION_RESOURCE_DIR)[i],
                       allow_pickle=True).flatten()
    demData = np.load(DEM_RESOURCE_DIR / pflh.getDirectoryFileNames(DEM_RESOURCE_DIR)[i],
                       allow_pickle=True).flatten()
    # tempData = np.load(CLIMATE_TEMPERATURE_RESOURCE_DIR / pflh.getDirectoryFileNames(CLIMATE_TEMPERATURE_RESOURCE_DIR)[i],
    #                    allow_pickle=True).flatten()

    # Load the projected coordinates
    projectionData = np.load(
        PROJECTIONS_DATASET_DIR / f"{resPrefix}_projectionCoords_resolutionScale_{resolution}_T{i + 1}.npy").flatten()

    offset = features * (count - 1 - i)
    for projectedIdx in range(len(projectionData)):
        # Get the index to look at
        dataIdx = int(projectionData[projectedIdx])
        precipValue = precData[dataIdx]
        demValue = demData[dataIdx]
        #tempValue = tempData[dataIdx]

        # Assign values to the climate dataset
        climateDataset[projectedIdx, offset] = precipValue
        climateDataset[projectedIdx, offset + 1] = demValue
        #climateDataset[projectedIdx, offset + 2] = tempValue


# Define column names
columns = []
# We have 4 features (precipitation, temperature, elevation, koppen) = 1 kernel
# [0.2,0.3,0.5,0.2] [0]
# [0.3,0.4,0.6,0.4] [1]
# [0.4,0.5,0.7,0.9] [2]
for i in range(count - 1, -1, -1):
    columns.append(f"Precipitation_T{i + 1}")
    columns.append(f"Elevation_T{i + 1}")
    #columns.append(f"Temperature_T{i + 1}")
    #columns.append(f"Koppen_T{i + 1}")
columns.append("SedimentaryLabel")

# View the dataframe to doublecheck we have populated columns
df = pd.DataFrame(climateDataset, columns=columns)
print(df.head())

cellFeatures = df.drop("SedimentaryLabel", axis=1).values
cellLabels = sedimentaryLabels.flatten()

# Log the number of true positives and true negatives
print("True Positives: ", np.sum(cellLabels == 1))
print("True Negatives: ", np.sum(cellLabels == 0))

# Normalise features
scaler = MinMaxScaler()
cellFeatures = scaler.fit_transform(cellFeatures)

# Reshape features into 3D (samples, timesteps, features)
cellFeatures = cellFeatures.reshape(cellFeatures.shape[0], count, features)

# Stratified split for train, test, and validation sets
xTrain, xVal, yTrain, yVal = train_test_split(cellFeatures, cellLabels, test_size=0.3, stratify=cellLabels, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, test_size=0.3, stratify=yVal, random_state=42)

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
model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(count, features)))
model.add(tf.keras.layers.LSTM(256, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.BatchNormalization())

# Middle LSTM Layers
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.LSTM(32, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

# Dense Layers for Final Classification
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
batchSize = 3750

# Check the model summary
model.summary()

# Train
history = model.fit(
    xTrain, yTrain,
    validation_data=(xVal, yVal),
    epochs=25,
    batch_size=batchSize,
    class_weight=classWeights,
    callbacks=[earlyStopping, lrScheduler],
    verbose=1
)

# Can load weights into the model if you want to test (Essentially loading a model)
#model.load_weights(pfl.MODELS_OUTPUT_DIR / "globalPredictions_Sedimentary_LSTM_0.9594.h5")

# Evaluate the model on the test set
testPredictions = model.predict(xTest).flatten()
# Only consider predictions with a 90% or higher confidence
testBinaryPredictions = (testPredictions >= 0.9).astype(int)

displayMetricsAgainstRandomGuessing(yTrain, yTest, testPredictions, testBinaryPredictions, "Sedimentary Test")

# Normalise and reshape global features
globalCellFeatures = scaler.transform(climateDataset[:, :-1])
globalCellFeatures = globalCellFeatures.reshape(globalCellFeatures.shape[0], count, features)
# Predict globally
globalPredictions = model.predict(globalCellFeatures).flatten()
globalBinaryPredictions = (globalPredictions >= 0.9).astype(int)

displayMetricsAgainstRandomGuessing(cellLabels, cellLabels, globalPredictions, globalBinaryPredictions, "Sedimentary Global")

# Create a DataFrame for global predictions
globalPredictionsDF = pd.DataFrame({
    "Predicted Sedimentary Suitability": globalPredictions,
    "Actual Sedimentary Labels": cellLabels.flatten()
})

# Save the prediction with the recall as a unique identifier
outputPath = pfl.MODELS_OUTPUT_DIR / f"globalPredictions_Sedimentary_LSTM_{accuracy_score(cellLabels, globalBinaryPredictions):.4f}.csv"
globalPredictionsDF.to_csv(outputPath, index=False)

# Save the weights too
outputPath = pfl.MODELS_OUTPUT_DIR / f"globalPredictions_Sedimentary_LSTM_{accuracy_score(cellLabels, globalBinaryPredictions):.4f}.h5"
model.save_weights(outputPath)