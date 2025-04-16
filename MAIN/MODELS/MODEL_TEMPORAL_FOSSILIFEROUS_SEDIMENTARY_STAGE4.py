import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from METRICS.METRIC_FEATURE_IMPORTANCE import displayFeatureImportance
from METRICS.METRIC_IOU import IoUScore
from METRICS.METRIC_SUMMARY_TABLES import displayMetricsAgainstRandomGuessing, getMetrics
from MODEL_TEMPORAL_LOGISTIC_REGRESSIONS import logisticModel_T1, logisticModel_FlatVector, logisticModel_MeanAverage
from VISUALISERS.PLOT_HARMONISATION_THRESHOLD import displayHarmonisedThresholdImpact
from VISUALISERS.PLOT_SPATIAL_THRESHOLD import displayThresholdImpact
from VISUALISERS.PLOT_TRAIN_HISTORY import plotAccuracy, plotLoss
from MODEL_KERASTUNER import BayesianBinaryOptim, HyperbandBinaryOptim
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This model is the final stage of the temporal aspect where we train against true ground truth labels on a transfer-learned model.
# Stage 3 is used with frozen layers to provide the fundamental underpinning of target regions for the new classifier to work with.

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
# Which period we should be looking to tune towards
geologicPrefix = "AllPeriods"

# Define the path to resources given by our input
resPrefix = f"{1 / resolution}x{1 / resolution}"

# FUNCTIONS -------------------------------------------------------------------

def transferModel():
    inputClimate = tf.keras.layers.Input(shape=(count, 3), name="Climate")
    inputHydro = tf.keras.layers.Input(shape=(count, 4), name="Hydro")
    inputErosion = tf.keras.layers.Input(shape=(count, 3), name="Erosion")

    # Freeze all LSTM layers
    lstmBiClimate = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, trainable=False))(
        inputClimate)
    lstmBiHydro = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(320, return_sequences=True, trainable=False))(
        inputHydro)
    lstmBiErosion = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, trainable=False))(
        inputErosion)
    lstmBiClimate = tf.keras.layers.BatchNormalization()(lstmBiClimate)
    lstmBiHydro = tf.keras.layers.BatchNormalization()(lstmBiHydro)
    lstmBiErosion = tf.keras.layers.BatchNormalization()(lstmBiErosion)
    lstmBiClimate = tf.keras.layers.Dropout(0.35)(lstmBiClimate)
    lstmBiHydro = tf.keras.layers.Dropout(0.05)(lstmBiHydro)
    lstmBiErosion = tf.keras.layers.Dropout(0.35)(lstmBiErosion)
    lstmBiClimate2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True, trainable=False))(
        lstmBiClimate)
    lstmBiHydro2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(160, return_sequences=True, trainable=False))(
        lstmBiHydro)
    lstmBiErosion2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True, trainable=False))(
        lstmBiErosion)
    lstmBiClimate2 = tf.keras.layers.BatchNormalization()(lstmBiClimate2)
    lstmBiHydro2 = tf.keras.layers.BatchNormalization()(lstmBiHydro2)
    lstmBiErosion2 = tf.keras.layers.BatchNormalization()(lstmBiErosion2)
    lstmBiClimate2 = tf.keras.layers.Dropout(0.3)(lstmBiClimate2)
    lstmBiHydro2 = tf.keras.layers.Dropout(0.1)(lstmBiHydro2)
    lstmBiErosion2 = tf.keras.layers.Dropout(0.3)(lstmBiErosion2)
    multiAttentionClimate = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64, trainable=False)(lstmBiClimate2,
                                                                                                         lstmBiClimate2)
    multiAttentionHydro = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64, trainable=False)(lstmBiHydro2,
                                                                                                       lstmBiHydro2)
    multiAttentionErosion = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64, trainable=False)(lstmBiErosion2,
                                                                                                         lstmBiErosion2)
    multiAttentionClimate = tf.keras.layers.Add()([multiAttentionClimate, lstmBiClimate2])
    multiAttentionHydro = tf.keras.layers.Add()([multiAttentionHydro, lstmBiHydro2])
    multiAttentionErosion = tf.keras.layers.Add()([multiAttentionErosion, lstmBiErosion2])
    multiAttentionClimate = tf.keras.layers.LayerNormalization()(multiAttentionClimate)
    multiAttentionHydro = tf.keras.layers.LayerNormalization()(multiAttentionHydro)
    multiAttentionErosion = tf.keras.layers.LayerNormalization()(multiAttentionErosion)
    flattenClimate = tf.keras.layers.GlobalAveragePooling1D()(multiAttentionClimate)
    flattenHydro = tf.keras.layers.GlobalAveragePooling1D()(multiAttentionHydro)
    flattenErosion = tf.keras.layers.GlobalAveragePooling1D()(multiAttentionErosion)
    concat = tf.keras.layers.concatenate([flattenClimate, flattenHydro, flattenErosion])
    concat = tf.keras.layers.BatchNormalization()(concat)
    concat = tf.keras.layers.Dropout(0.1)(concat)

    # Final Dense Layers
    dense1 = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005))(concat)
    dense1 = tf.keras.layers.Dropout(0.1)(dense1)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dense1)

    model = keras.models.Model(inputs=[inputClimate, inputHydro, inputErosion], outputs=output)
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
PROJECTIONS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS"
FOSSIL_SEDIMENT_LABELS_DIR = pfl.DATASET_DIR / f"{resPrefix}SedimentaryRockCategories"
FOSSIL_TRUTH_LABELS_DIR = pfl.DATASET_DIR / f"{resPrefix}FossilOccurrences"
SPATIAL_MODEL_PREDICTIONS_DIR = pfl.PREDICTIONS_DIR / f"{resPrefix}SpatialModel"
COMPILED_DATAFRAMES_DIR = pfl.DATASET_DIR / f"CompiledDataFrames"
pflh.createDirectoryIfNotExist(COMPILED_DATAFRAMES_DIR)
MODEL_PREDICTIONS_DIR = pfl.PREDICTIONS_DIR / f"{resPrefix}TemporalModel"
pflh.createDirectoryIfNotExist(MODEL_PREDICTIONS_DIR)

# Define column names (features names)
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
columns.append("FossiliferousLabel")

if not loadCompiledDataframe:
    # Mask to ignore any ocean or ice cells and focus on terrestrial cells
    oceanMask = np.load(FOSSIL_SEDIMENT_LABELS_DIR / f"OceanMaskBinary.npy", allow_pickle=True).flatten()
    # Load the fossiliferous labels
    geologicFocusedLabels = np.load(FOSSIL_TRUTH_LABELS_DIR / f"{geologicPrefix.upper()}_FULL_OCCURRENCES.npy", allow_pickle=True).flatten()

    # count * X features + 1 label
    # 'climate' refers to the entire dataset, not to be confused with climate variables
    climateDataset = np.zeros(((180 * resolution + 1) * (360 * resolution + 1), (count * features) + 1))
    climateDataset[:, -1] = geologicFocusedLabels.flatten()

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
    np.save(COMPILED_DATAFRAMES_DIR / f"{resPrefix}{geologicPrefix}Fossiliferous(STAGE4)Dataset.npy", df.values)

# Load the dataframe from the .npy file
df = pd.DataFrame(np.load(COMPILED_DATAFRAMES_DIR / f"{resPrefix}{geologicPrefix}Fossiliferous(STAGE4)Dataset.npy"), columns=columns)
print(df.head())
cellLabels = df["FossiliferousLabel"].values.flatten()
cellFeatures = df.drop("FossiliferousLabel", axis=1).values

# Load the recall boost spatial model predictions
recallBoostPredictions = np.load(SPATIAL_MODEL_PREDICTIONS_DIR / f"ALLPERIODS_RECALL_BOOST.npy", allow_pickle=True).flatten()

# Get indices for where we know there has been fossil activity
trueSamples = np.where(cellLabels == 1)[0]
# Get all the other indices where there's not been any reported fossil activity relative to the geologic period
negativeSamples = np.where(cellLabels == 0)[0]
# Randomly choose from the negative group to satiate the same true negatives for the model on every run
np.random.seed(42)
# Currently trying not to oversample negatives to prevent the model from catering against the true positives
randomNegativeSampledIndices = np.random.choice(negativeSamples, min(2 * len(trueSamples), len(negativeSamples)), replace=False)
# Combine true and false
allSamplesIndices = np.concatenate([trueSamples, randomNegativeSampledIndices])
# Update our labels and features with the new subset
cellLabels = cellLabels[allSamplesIndices]
cellFeatures = cellFeatures[allSamplesIndices]
cellRecallBoostLabels = recallBoostPredictions[allSamplesIndices]

# Log the number of true positives and true negatives
print("True Positives: ", np.sum(cellLabels == 1))
print("True Negatives: ", np.sum(cellLabels == 0))

# Find the feature importance of the dataset
# displayFeatureImportance(cellFeatures, cellLabels, df, "FossiliferousLabel")

# Stratified split for train/test/val sets
xTrain, xVal, yTrain, yVal = train_test_split(cellFeatures, cellLabels, test_size=0.3, stratify=cellLabels, shuffle=True, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, test_size=0.6, stratify=yVal, shuffle=True, random_state=42)

# Compute class weights
classWeights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(yTrain),
    y=yTrain
)
# Convert to dictionary
classWeights = dict(enumerate(classWeights))

# Use the standard scaler
scaler = StandardScaler()
# Flatten to features and then reshape to the regular format for RNN
xTrain = scaler.fit_transform(xTrain.reshape(-1, features)).reshape(-1, count, features)
xVal = scaler.transform(xVal.reshape(-1, features)).reshape(-1, count, features)
xTest = scaler.transform(xTest.reshape(-1, features)).reshape(-1, count, features)

# (OPTIONAL) Run preliminary logistic regression models to see if there's any temporal importance in the dataset
# logisticModel_T1(xTrain, yTrain, xTest, yTest, 0.90,f"{geologicPrefix} Fossiliferous")
# logisticModel_FlatVector(xTrain, yTrain, xTest, yTest, 0.90,f"{geologicPrefix} Fossiliferous")
# logisticModel_MeanAverage(xTrain, yTrain, xTest, yTest, 0.90, f"{geologicPrefix} Fossiliferous")

# Compile the model
tf.keras.backend.clear_session()
model = transferModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Load weights from previous stage for transfer-learning
model.load_weights(pfl.MODELS_OUTPUT_DIR / "testPredictions_categoricalAgedSedimentary_MULTIHEAD_LSTM_0.8650.h5",
                   by_name=True, skip_mismatch=True)

xTrainClimate = xTrain[:, :, :3]
xTrainHydro = xTrain[:, :, 3:7]
xTrainErosion = xTrain[:, :, 7:10]
# Validation feature groups
xValClimate = xVal[:, :, :3]
xValHydro = xVal[:, :, 3:7]
xValErosion = xVal[:, :, 7:10]

# Check the model summary
model.summary()

# Train
# history = model.fit(
#     [xTrainClimate, xTrainHydro, xTrainErosion], yTrain,
#     validation_data=([xValClimate, xValHydro, xValErosion], yVal),
#     epochs=85,
#     batch_size=256,
#     class_weight=classWeights,
#     verbose=1
# )
#
# plotAccuracy(history)
# plotLoss(history)

# Begin KFold on dataset for transfer-learned model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kFoldAcc = []
kFoldLoss = []
kFoldPrec = []
kFoldRec = []
kFoldF1 = []
kFoldConf = []
kFoldAUCROC = []
globalPredictions = np.zeros(len(cellLabels))
globalPredictionsRaw = np.zeros(len(cellLabels))
for fold, (trainIdx, valIdx) in enumerate(skf.split(cellFeatures, cellLabels)):
    print(f"\nComputing Fold: {fold + 1} / {5}")

    # Split data into val and train
    xTrain, xVal = cellFeatures[trainIdx], cellFeatures[valIdx]
    yTrain, yVal = cellLabels[trainIdx], cellLabels[valIdx]

    # Compute class weights
    classWeights = compute_class_weight(class_weight="balanced", classes=np.unique(yTrain), y=yTrain)
    classWeights = dict(enumerate(classWeights))

    # Scale the data
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain.reshape(-1, features)).reshape(-1, count, features)
    xVal = scaler.transform(xVal.reshape(-1, features)).reshape(-1, count, features)

    # Compile the model
    tf.keras.backend.clear_session()
    model = transferModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Load weights from previous stage for transfer-learning
    model.load_weights(pfl.MODELS_OUTPUT_DIR / "testPredictions_categoricalAgedSedimentary_MULTIHEAD_LSTM_0.8650.h5", by_name=True, skip_mismatch=True)

    xTrainClimate = xTrain[:, :, :3]
    xTrainHydro = xTrain[:, :, 3:7]
    xTrainErosion = xTrain[:, :, 7:10]
    # Validation feature groups
    xValClimate = xVal[:, :, :3]
    xValHydro = xVal[:, :, 3:7]
    xValErosion = xVal[:, :, 7:10]

    # Train
    history = model.fit(
        [xTrainClimate, xTrainHydro, xTrainErosion], yTrain,
        validation_data=([xValClimate, xValHydro, xValErosion], yVal),
        epochs=85,
        batch_size=256,
        class_weight=classWeights,
        verbose=1
    )

    # Save results
    kFoldAcc.append(max(history.history["val_accuracy"]))
    kFoldLoss.append(min(history.history["val_loss"]))

    # Generate predictions from validation set
    valPredictions = model.predict([xValClimate, xValHydro, xValErosion]).flatten()
    valBinaryPredictions = (valPredictions >= 0.90).astype(int)

    # Store the predictions in their correct index locations
    globalPredictions[valIdx] = valBinaryPredictions
    globalPredictionsRaw[valIdx] = valPredictions

    prec, rec, f1, conf, AUCROC = getMetrics(yVal, valPredictions, valBinaryPredictions, "Predictions")
    kFoldPrec.append(prec)
    kFoldRec.append(rec)
    kFoldF1.append(f1)
    kFoldConf.append(conf)
    kFoldAUCROC.append(AUCROC)

# Print mean results across all folds
print(f"{geologicPrefix} Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(kFoldAcc):.4f}")
print(f"Mean Loss: {np.mean(kFoldLoss):.4f}")
print(f"----------------------------")
print(f"Mean Precision: {np.mean(kFoldPrec):.4f}")
print(f"Mean Recall: {np.mean(kFoldRec):.4f}")
print(f"Mean F1: {np.mean(kFoldF1):.4f}")
print(f"Mean Fossiliferous Confidence: {np.mean(kFoldConf):.4f}")
print(f"Mean AUC-ROC: {np.mean(kFoldAUCROC):.4f}")

# Print the metrics of the adjusted predictions
prec, rec, f1, conf, AUCROC = getMetrics(cellLabels, globalPredictionsRaw, globalPredictions, "Temporal Only Predictions")
# Save the globalRaw
np.save(MODEL_PREDICTIONS_DIR / f"{geologicPrefix}TemporalOnlyPredictions.npy", globalPredictionsRaw)

# Load the saved values
#globalPredictionsRaw = np.load(MODEL_PREDICTIONS_DIR / f"{geologicPrefix}TemporalOnlyPredictions.npy", allow_pickle=True).flatten()

# Find an optimal point
displayHarmonisedThresholdImpact(cellLabels, globalPredictionsRaw, cellRecallBoostLabels, 0.0, 0.5, 0.01)

# Apply the recall boosted values elementwise
adjustedPredictions = np.copy(globalPredictionsRaw)
# For every cell that has been boosted by the spatial model, increase the prediction by the harmonisation weight
adjustedPredictions[cellRecallBoostLabels == 1] += 0.4
# Ensure the adjusted predictions do not exceed the maximum of 1
adjustedPredictions = np.clip(adjustedPredictions, 0, 1)
adjustedBinaryPredictions = (adjustedPredictions >= 0.90).astype(int)
# Print the metrics of the adjusted predictions
prec, rec, f1, conf, AUCROC = getMetrics(cellLabels, adjustedPredictions, adjustedBinaryPredictions, "Harmonised Predictions")

print("IoU Score: ", IoUScore(cellLabels.flatten(), adjustedBinaryPredictions.flatten()))
