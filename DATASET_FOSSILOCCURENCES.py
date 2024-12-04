import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import PFL_HELPER as pflh
import PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# Create a directory for the Fossils
FOSSIL_DIR = pfl.DATASET_DIR / "FOSSIL"
pflh.createDirectoryIfNotExist(FOSSIL_DIR)

# Define the path to fossil resources
FOSSIL_RESOURCE_DIR = pfl.RESOURCES_DIR / "FossilRecord"

# Specify the resolution, technically 'infinite' and therefore we can downscale to any resolution
# 1 = 1x1 degree, 2 = 0.5x0.5 degree, 5 = 0.2x0.2 degree, 10 = 0.1x0.1 degree
resolutionScale = 1

# Iterate through the count times to produce the fossil projections across the count * timestep time period
fossilTarget = np.zeros((180 * resolutionScale + 1) * (360 * resolutionScale + 1))
# Get a file by specific index for now
file = pflh.getDirectoryFileNames(FOSSIL_RESOURCE_DIR)[0]
with open(os.path.join(FOSSIL_RESOURCE_DIR, file), "r", encoding="utf-8") as f:
    data = f.read()

# Cleanup the existing CSV data and craft 2 arrays for longitude and latitude
rows = data.split("\n")[20:-1]
fossilLongs = []
fossilLats = []
for i in tqdm(range(0, len(rows)), desc="Cleaning Fossil Coordinates"):
    long = (float(rows[i].split(",")[17].replace('"', "")))
    lat = (float(rows[i].split(",")[18].replace('"', "")))
    # Ensure the longitude and latitude are within the valid range
    if -180 <= long < 180 and -90 <= lat <= 90:
        fossilLongs.append(long)
        fossilLats.append(lat)

# print(projected_long)
for i in tqdm(range(0, len(fossilLongs)), desc="Projecting Fossil Coordinates"):
    # Convert the projected longitude and latitude to grid indices
    longIdx = int((fossilLongs[i] + 180) * resolutionScale)
    latIdx = int((90 - fossilLats[i]) * resolutionScale)

    # Ensure indices are within bounds
    if 0 <= longIdx < 360 * resolutionScale and 0 <= latIdx < 180 * resolutionScale:
        # Find the index from the flattened target and increment it, clamped at "1" to avoid any sampling bias and also for binary classification
        fossilTarget[latIdx * (360 * resolutionScale + 1) + longIdx] = min(1, fossilTarget[latIdx * (360 * resolutionScale + 1) + longIdx] + 1)

# Output file to the right directory and name
outDir = FOSSIL_DIR / f"{file.split('.')[0]}_resolutionScale_{resolutionScale}_occurences_{int(np.sum(fossilTarget))}"
pflh.createDirectoryIfNotExist(outDir)
outFile = outDir / f"{file.split('.')[0]}_resolutionScale_{resolutionScale}_occurences_{int(np.sum(fossilTarget))}.npy"
if not os.path.isfile(outFile):
    np.save(outFile, fossilTarget)

# Check how many positive labels we have
print(f"Total Fossil Occurrences: {np.sum(fossilTarget)}")

plt.figure(figsize=(20, 10))
sc = plt.imshow(fossilTarget.reshape((180 * resolutionScale + 1), (360 * resolutionScale + 1)), cmap="hot", interpolation="nearest")
plt.title(label=f"Fossil Occurrences at resolution={resolutionScale}")
plt.colorbar(sc, label="Occurrences")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)
plt.show()
