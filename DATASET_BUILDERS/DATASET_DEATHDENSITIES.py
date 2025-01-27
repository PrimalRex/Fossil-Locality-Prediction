import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from MAIN import GPLATES_ROTATION_MODEL as gprm
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This script generates death densities based on occurrences from each file in the FOSSIL_TIMESTEP_DIR
# Utilises the same logic as DATASET_FOSSILOCCURENCES... May be merged in a future push

# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 4
steps = 5

resPrefix = f"{1 / resolution}x{1 / resolution}"
DEM_RESOURCE_DIR = pfl.RESOURCES_DIR / f"{resPrefix}DEMs"
# TODO: REMOVE DEPENDENCY ON THIS SPECIFIC DIRECTORY FOR FOSSIL OCCURRENCES
FOSSIL_TIMESTEP_DIR = pfl.DATASET_DIR / "FOSSIL" / "245_0_FOSSIL_OCCURENCES_BY_5_TIMESTEP"
PROJECTIONS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS"
projectionData = np.load(PROJECTIONS_DATASET_DIR / f"{resPrefix}_projectionCoords_resolutionScale_{resolution}_T{50}.npy")

GIS_FOSSIL_TIMESTEP_DIR = pfl.DATASET_DIR / "FOSSIL" / "245_0_FOSSIL_OCCURENCES_BY_5_TIMESTEP_GIS"
pflh.createDirectoryIfNotExist(GIS_FOSSIL_TIMESTEP_DIR)

# Create a list of all combinations of longitudes and latitudes
longitudes, latitudes = np.meshgrid(np.arange(-180, 180 + (1 / resolution), 1 / resolution),
                                    np.arange(90, -90 - (1 / resolution), -1 / resolution))

# Create all permutations of longitudes and latitudes
coordPairs = np.column_stack((longitudes.ravel(), latitudes.ravel()))

for i in range (0, pflh.getDirectoryFileCount(FOSSIL_TIMESTEP_DIR)):
    # Iterate through the count times to produce the fossil projections across the count * timestep time period
    fossilTarget = np.zeros((180 * resolution + 1) * (360 * resolution + 1))
    # Get a file by specific index for now
    file = pflh.getDirectoryFileNames(FOSSIL_TIMESTEP_DIR)[i]
    with open(os.path.join(FOSSIL_TIMESTEP_DIR, file), "r", encoding="utf-8") as f:
        data = f.read()

    print(pflh.getDirectoryFileNames(FOSSIL_TIMESTEP_DIR)[i])
    # Cleanup the existing CSV data and craft 2 arrays for longitude and latitude
    rows = data.split("\n")[21:-1]
    fossilLongs = []
    fossilLats = []
    for j in tqdm(range(0, len(rows)), desc="Cleaning Fossil Coordinates"):
        try:
            long = (float(rows[j].split(",")[17].replace('"', "")))
            lat = (float(rows[j].split(",")[18].replace('"', "")))
            # Ensure the longitude and latitude are within the valid range
            if -180 <= long < 180 and -90 <= lat <= 90:
                fossilLongs.append(long)
                fossilLats.append(lat)
        except:
            continue

    # Project the fossil coordinates to the current time slice
    fossilLongs, fossilLats = gprm.getPalaeoCoordinates(i * steps, fossilLongs, fossilLats)
    # print(projected_long)
    dataList = []
    for k in tqdm(range(0, len(fossilLongs)), desc="Projecting Fossil Coordinates"):
        # Convert the projected longitude and latitude to grid indices
        longIdx = int((fossilLongs[k] + 180) * resolution)
        latIdx = int((90 - fossilLats[k]) * resolution)

        # Ensure indices are within bounds
        if 0 <= longIdx < 360 * resolution and 0 <= latIdx < 180 * resolution:
            # Find the index from the flattened target and increment it by 0.1, we clamp at 1 to normalize the density
            # Attempting to use sampling bias to our advantage here in order to prove density prediction
            fossilTarget[latIdx * (360 * resolution + 1) + longIdx] = min(5, fossilTarget[latIdx * (360 * resolution + 1) + longIdx] + 1)

    # Visualise each map if we have any occurrences
    print(f"Total Fossil Occurrences: {np.sum(fossilTarget)}")
    if np.sum(fossilTarget) > 0:
        plt.figure(figsize=(20, 10))
        sc = plt.imshow(fossilTarget.reshape((180 * resolution + 1), (360 * resolution + 1)), cmap="hot", interpolation="nearest")
        plt.title(label=f"{file} Fossil Occurrences at resolution={resolution}")
        #plt.colorbar(sc, label="Occurrences")
        #plt.xlabel("Longitude")
        #plt.ylabel("Latitude")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    # Save the death density to a CSV file
    combinedData = np.column_stack((coordPairs, fossilTarget))
    fossilDF = pd.DataFrame(combinedData, columns=["Longitude", "Latitude", "Value"])
    outPath = os.path.join(GIS_FOSSIL_TIMESTEP_DIR, f"{file}_GIS.csv")
    fossilDF.to_csv(outPath, index=False, header=False)
