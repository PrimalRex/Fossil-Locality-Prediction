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

BINNED_FOSSILS_DIR = pfl.DATASET_DIR / "FOSSIL_OCCURRENCE_ANALYSIS" / f"BINNED_BY_{steps}Myr"
DEATHDENSITY_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}DeathDensities"
pflh.createDirectoryIfNotExist(DEATHDENSITY_DATASET_DIR)

mainRasterTarget = np.zeros((180 * resolution + 1) * (360 * resolution + 1))

for i in reversed(range(0, len(pflh.getDirectoryFileNames(BINNED_FOSSILS_DIR)))):
    # Iterate through the count times to produce the fossil projections across the count * timestep time period
    fossilTarget = np.zeros((180 * resolution + 1) * (360 * resolution + 1))
    # Get a file by specific index for now
    file = pflh.getDirectoryFileNames(BINNED_FOSSILS_DIR)[i]
    with open(os.path.join(BINNED_FOSSILS_DIR, file), "r", encoding="utf-8") as f:
        data = f.read()

    print(pflh.getDirectoryFileNames(BINNED_FOSSILS_DIR)[i])
    # Cleanup the existing CSV data and craft 2 arrays for longitude and latitude
    rows = data.split("\n")[1:-1]
    fossilLongs = []
    fossilLats = []
    for j in tqdm(range(0, len(rows)), desc="Cleaning Fossil Coordinates"):
        try:
            long = (float(rows[j].split(",")[2].replace('"', "")))
            lat = (float(rows[j].split(",")[3].replace('"', "")))
            # Ensure the longitude and latitude are within the valid range
            if -180 <= long < 180 and -90 <= lat <= 90:
                fossilLongs.append(long)
                fossilLats.append(lat)
        except:
            continue

    # Project the fossil coordinates to the current time slice
    if len(fossilLongs) > 0:
        fossilLongs, fossilLats = gprm.getPalaeoCoordinates(i * steps, fossilLongs, fossilLats, "paleomap")
        # print(projected_long)
        dataList = []
        for k in tqdm(range(0, len(fossilLongs)), desc="Projecting Fossil Coordinates"):
            # Convert the projected longitude and latitude to grid indices
            longIdx = int((fossilLongs[k] + 180) * resolution)
            latIdx = int((90 - fossilLats[k]) * resolution)

            # Ensure indices are within bounds
            if 0 <= longIdx < 360 * resolution and 0 <= latIdx < 180 * resolution:
                # Find the index from the flattened target and increment it by 1 we clamp at 5 to normalise the density
                # Attempting to use sampling bias to our advantage here in order to prove density prediction
                fossilTarget[latIdx * (360 * resolution + 1) + longIdx] = min(5, fossilTarget[latIdx * (360 * resolution + 1) + longIdx] + 1)

    # Our decay function can be variate, higher figures means there's less decay
    mainRasterTarget *= 0.0
    # Add the current fossil target to the main raster target
    mainRasterTarget += fossilTarget

    # Visualise each map if we have any occurrences
    print(f"Total Fossil Occurrences: {np.sum(mainRasterTarget)}")
    if np.sum(fossilTarget) > 0:
        plt.figure(figsize=(20, 10))
        sc = plt.imshow(mainRasterTarget.reshape((180 * resolution + 1), (360 * resolution + 1)), cmap="hot", interpolation="nearest")
        plt.title(label=f"{file} Fossil Death Densities at resolution={resolution}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    # Save the timestep state of the main target as an NPY file
    np.save(os.path.join(DEATHDENSITY_DATASET_DIR, f"{file}_deathdensitydecay_{(360 * resolution + 1)}.npy"), fossilTarget)
