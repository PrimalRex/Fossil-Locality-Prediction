![image](https://github.com/user-attachments/assets/c4c5fd41-837d-4f3e-b477-6e6c8642752c)

> Machine Learning on Palaeogeography & Climatology (BSc Thesis)

----

**PFL** is a python-based machine learning system which combines palaeographic data to predict period-specific regions of fossiliferous significance.

----

![OutputApproxAgeMap_Raster](https://github.com/user-attachments/assets/e62a29fa-7ae5-470e-8873-404719f4b99a)

```
CAPTION: [BEDROCK BY OCCURENCES]
```

## 🌎 Overview of System
What is it? - It is a temporal and spatial prediction study. I took a series of features and evaluated against a set of lithological regions and fossil occurences. I produced final outputs by superimposing spatial on temporal predictions.

I explored deep-time data, subsequently built time-aware models to facilitate our predictions. In addition, I produced predictions using a singular time-step to investigate contemporaneous influence. The fundamental experimentation was driven through the following pipeline:

![image](https://github.com/user-attachments/assets/03ab25f5-13cf-43e5-9c42-243b354d7e81)

### ⌛ Temporal Take-Aways
- Beat all Logistic Models and Random Guessing consistently.
- Demonstrated high metrics in sedimentary landmass recognition.
- Understood preservation relationships to determine pseudo-ages of sedimentary rock.

### 🗺 Spatial Take-Aways
![image](https://github.com/user-attachments/assets/ceb40562-2af5-4f4a-8d53-1dcd52e84d60)
- Produced a K-Fold Cross-Validated global prediction output.
- Used virtually all elected temporal features, demonstrating consistency and validity in the study.

  
### 📈 Results
![image](https://github.com/user-attachments/assets/6260687b-ee50-43b5-9cef-ef4a776ec9f8)
- Precisions demonstrate strong positive prediction rates, ensuring accurate predictions.
- IoUs demonstrate respectable spatial relationships, proving against the notion of sporadic guessful predictions.
- AUC-ROCs showing appropriate model usage.



## 🧪 Repository Overview
### ANALYSIS
The [directory](https://github.com/PrimalRex/Fossil-Locality-Prediction/tree/main/ANALYSIS) contains two python scripts, both pertain to cleaning of fossil occurrence data. 

The [first](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/ANALYSIS/ALL_FOSSILS_TO_TIMESTEP.py) tool is used to aggregate a set of cleaned occurences into timestep binning. I ended up using this for one specific feature in our investigation: 'Death Signals'.

The [second](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/ANALYSIS/PALEODB_FOSSIL_ANALYSIS.py) tool is used in cleaning reconnaissance that I obtained from [PaleoBioDB](https://paleobiodb.org). It reads the API response CSVs, cleans entries based on, coordinate, entry No.'s, min-max age, environments and assigned temporal bins. Additionally it prints before and after quantities for this imputation. 
This is an essential script to clean the data, it is function built to operate with whichever queried data as long as they match these parameters:

| Parameter  | Example |
| ------------- | ------------- |
| max_ma  | 66 |
| min_ma  | 23 |
| envtype  | !marine / marine |
| FILENAME | 023_066_XXX.csv|
```
http://paleobiodb.org/data1.2/occs/list.csv?datainfo&rowcount&max_ma=XXX&min_ma=XXX&lithology=!other,metasedimentary,metamorphic,volcanic,unknown&envtype=XXX&show=coords,env,timebins
```

### DATASET_BUILDERS
This directory contains scripts that I used to build our .NPY datasets, they are multi-variate and are commented appropriately. The following are brief summaries of each file.
| File  | Summary |
| ------------- | ------------- |
|[DATASET_0.1X0.1DEM_NC_TO_NPY.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/DATASET_BUILDERS/DATASET_0.1X0.1DEM_NC_TO_NPY.py)| Used to convert .netCDF files, specifically palaeographic NC files, into a .NPY file. Useful for evaluating raw data that doesn't need any imputation or cleaning.
|[DATASET_0.1x0.1DEM_NC_TO_CSV.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/DATASET_BUILDERS/DATASET_0.1x0.1DEM_NC_TO_CSV.py)| Similar to the previous file but converts to .CSV, ready to be read into GIS software.
|[DATASET_COORD_PROJECTIONS.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/DATASET_BUILDERS/DATASET_COORD_PROJECTIONS.py)| Used to produce projection matrices, frequently used to ensure cellular time-step consistency due to drastic plate movement in earth's history.
|[DATASET_DEATHDENSITIES.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/DATASET_BUILDERS/DATASET_DEATHDENSITIES.py)| Candidate feature 'Death Signal', found little influence in its given parameter setup, has support for signal decay.
|[DATASET_FOSSILOCCURENCES.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/DATASET_BUILDERS/DATASET_FOSSILOCCURENCES.py)| Used to convert cleaned occurrence CSVs into appropriate spatial grids for .NPY format. No projections are applied, has support for clamping densities per cell.
|[DATASET_GISEXPORTED_CSV_TO_NPY.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/DATASET_BUILDERS/DATASET_GISEXPORTED_CSV_TO_NPY.py)| Used to convert GIS exported data into .NPYs, batch converts all files in a folder for quick and efficient coversion for temporal datasets.
|[DATASET_PALEODB_DOWNLOAD_MANAGER.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/DATASET_BUILDERS/DATASET_PALEODB_DOWNLOAD_MANAGER.py)| DEPRICATED - But still may be of potential use to streamline reconnaissance due to server time-outs from PaleoBioDB.

### MAIN/
This [directory](https://github.com/PrimalRex/Fossil-Locality-Prediction/tree/main/MAIN) is the root of the main machine-learning models and primary files. The following are summaries of the root files, these are referenced as packaged in other files.
| File  | Summary |
| ------------- | ------------- |
|[GPLATES_ROTATION_MODEL.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/GPLATES_ROTATION_MODEL.py)| A clean module for the GPlately package to streamline coordinate projection invocation.
|[PFL_GPU_ENABLETEST.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/PFL_GPU_ENABLETEST.py)| Simple test to ensure cuDNN functionality.
|[PFL_HELPER.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/PFL_HELPER.py)| Helper directory functions.
|[PFL_PATHS.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/PFL_PATHS.py)| Helper directory names.

### MAIN/MODELS
This [directory](https://github.com/PrimalRex/Fossil-Locality-Prediction/tree/main/MAIN/MODELS) contains all model files. Each file has apt descriptions of their usage relative to the written report, however I will briefly mention the various predictions I made:
| File  | Summary |
| ------------- | ------------- |
|[MODEL_TEMPORAL_LOGISTIC_REGRESSIONS.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/MODELS/MODEL_TEMPORAL_LOGISTIC_REGRESSIONS.py)| Logistic Regression Models that focus on T(1), T(AVG), T(FLAT) models to validate temporal ordinance and signficance in the study.
|[MODEL_SPATIAL_FOSSILIFEROUS_RECALLBOOST.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/MODELS/MODEL_SPATIAL_FOSSILIFEROUS_RECALLBOOST.py)| Spatial Model using 0 MA data to produce fossil discovery potential spots.
|[MODEL_TEMPORAL_SEDIMENTARY_STAGE2_LSTM.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/MODELS/MODEL_TEMPORAL_SEDIMENTARY_STAGE2_LSTM.py)| An LSTM (RNN) Model to predict regions of sedimentary vs. non-sedimentary across a temporal frame at fixed time intervals.
|[MODEL_TEMPORAL_AGED_SEDIMENTARY_STAGE3_LSTM.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/MODELS/MODEL_TEMPORAL_AGED_SEDIMENTARY_STAGE3_LSTM.py)| An LSTM (RNN) Model to predict the age of given sedimentary regions, ages are derived via fossil overlap.
|[MODEL_TEMPORAL_AGED_SEDIMENTARY_STAGE3_MULTIHEAD_LSTM.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/MODELS/MODEL_TEMPORAL_AGED_SEDIMENTARY_STAGE3_MULTIHEAD_LSTM.py)| Same as the above however utilising a multi-head model to segregate features into distinct groups.
|[MODEL_TEMPORAL_FOSSILIFEROUS_SEDIMENTARY_STAGE4.py](https://github.com/PrimalRex/Fossil-Locality-Prediction/blob/main/MAIN/MODELS/MODEL_TEMPORAL_FOSSILIFEROUS_SEDIMENTARY_STAGE4.py)| Final RNN Model where the previous multi-head model is used for transfer-learning, additionally the spatial model is used to produce the harmonic output.

### METRICS
This [directory](https://github.com/PrimalRex/Fossil-Locality-Prediction/tree/main/METRICS) contains methods for printing and calculating various metrics, specifically IOU and Confidence.

### VISUALISERS
Lastly, this [directory](https://github.com/PrimalRex/Fossil-Locality-Prediction/tree/main/VISUALISERS) contains some helpful graph plotting functions.


## 🛠 Prerequisites
This project was built with a certain version of Python to ensure holistic compatibility for all required modules and GPU Acceleration.

| Package  | Version |
| :---:  | :---: |
| Python  | 3.10.15 |
| TensorFlow  | 2.10.1 |
| cuDNN (OPTIONAL)  | 11.2.x |
| CUDA (OPTIONAL)  | 8.1.x |

You can view the compatibility list [here](https://www.tensorflow.org/install/source#gpu) for GPU Acceleration support on different Python versions.

### 📦 Datasets & Files
You can either download all or individual distribution-ready packages from the [releases](https://github.com/PrimalRex/Fossil-Locality-Prediction/releases/tag/v1.0-release) tab.
