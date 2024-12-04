import math
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import PFL_HELPER as pflh
import PFL_PATHS as pfl

# WARNING - PENDING DEPRECATION: This script will process the Mean Annual Precipitation and Temperature data and produce a temporal composite
# of the data. It can be controlled via alpha and decay to give more emphasis on the temporal timeline however the benefits of this composite
# is still being researched - this is more of a proof of concept and a demonstration for loading/processing/previewing data.

# FUNCTIONS -------------------------------------------------------------------

def processClimateResources(CLIMATE_RESOURCE_DIR, alpha, shift, climateType):
    # Get the file count
    fileCount = pflh.getDirectoryFileCount(CLIMATE_RESOURCE_DIR)
    #fileCount = 1
    # Initialize the sum of intensities to the length of the dataset
    sumIntensities = np.zeros(181 * 361)
    # Time period tags
    start = pflh.getDirectoryFileNames(CLIMATE_RESOURCE_DIR)[0].split("_")[0]
    end = pflh.getDirectoryFileNames(CLIMATE_RESOURCE_DIR)[-1].split("_")[0]

    # Iterate through all the files in the directory
    for file, filename in tqdm(enumerate(os.listdir(CLIMATE_RESOURCE_DIR)), total=fileCount,
                               desc=f"Building Composite {climateType}"):
        # Read the data from the file
        with open(os.path.join(CLIMATE_RESOURCE_DIR, filename), "r") as f:
            data = f.read()
        rows = data.split("\n")[1:-1]

        for i in range(0, len(rows)):
            # Calculate the sum intensity for each coordinate factoring decay and shift
            sumIntensities[i] += (float(rows[i].split(",")[2]) / fileCount) * alpha ** (max(0, file - shift))

    outFile = CLIMATE_DIR / f"{start}_{end}_sum_{climateType}_decay_{alpha}_shift_{shift}.npy"
    if not os.path.isfile(outFile):
        np.save(outFile, sumIntensities)
    return sumIntensities, start, end


def plotClimateData(intensities, start, end, alpha, shift, climateType, cmap):
    # Plot the data using a scatter graph
    plt.figure(figsize=(20, 10))
    sc = plt.imshow(intensities.reshape(181, 361), cmap=cmap, interpolation="nearest")
    plt.colorbar(sc, label=f"{climateType}")
    plt.title(f"{start} - {end} Mean Annual {climateType}, decay={alpha}, shift={shift}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(False)
    plt.show()


# MAIN ------------------------------------------------------------------------

# Create a directory for the Climate if it doesn't exist
CLIMATE_DIR = pfl.DATASET_DIR / "CLIMATE"
pflh.createDirectoryIfNotExist(CLIMATE_DIR)

# Define the path to resources (Precipitation and Temperatures)
CLIMATE_PRECIPITATION_RESOURCE_DIR = pfl.RESOURCES_DIR / "1.0x1.0MeanAnnualPrecipitation"
CLIMATE_TEMPERATURE_RESOURCE_DIR = pfl.RESOURCES_DIR / "1.0x1.0MeanAnnualTemperatures"

# Decay Rate: 0.1 = Extreme Decay, 1 = Equal Weighted Average
alpha = .3
# Shift the decay to focus on a specific time period
shift = math.floor(pflh.getDirectoryFileCount(CLIMATE_PRECIPITATION_RESOURCE_DIR) * .14)

# Process precipitation files
print("Processing Precipitation Data...")
precipitationIntensities, start, end = processClimateResources(CLIMATE_PRECIPITATION_RESOURCE_DIR, alpha, shift,
                                                               "Precipitation")
plotClimateData(precipitationIntensities, start, end, alpha, shift, "Precipitation", "viridis")

alpha = .3
shift = math.floor(pflh.getDirectoryFileCount(CLIMATE_TEMPERATURE_RESOURCE_DIR) * .14)

# Process temperature files
print("Processing Temperature Data...")
temperatureIntensities, start, end = processClimateResources(CLIMATE_TEMPERATURE_RESOURCE_DIR, alpha, shift,
                                                             "Temperature")
plotClimateData(temperatureIntensities, start, end, alpha, shift, "Temperature", "hot")
