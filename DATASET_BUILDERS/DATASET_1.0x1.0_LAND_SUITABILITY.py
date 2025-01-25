import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This script generates a ground truth map for true negatives based on DEM data of the projected time interval
# and the current earth suitability, for example, oceans are unsuitable for any fossil densities.
# This is purely to aid training by helping the model associate the right labels with the right patterns.
# TODO: This script is not yet finished, and is still in prototype and research stage.

# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 1

resPrefix = f"{1 / resolution}x{1 / resolution}"
DEM_RESOURCE_DIR = pfl.RESOURCES_DIR / f"{resPrefix}DEMs"
PROJECTIONS_DATASET_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS"
OCEANMASK_RESOURCE_DIR = pfl.RESOURCES_DIR / f"{resPrefix}OceanMask"

projectionData = np.load(PROJECTIONS_DATASET_DIR / f"{resPrefix}_projectionCoords_resolutionScale_{resolution}_T{50}.npy")

with open(os.path.join(DEM_RESOURCE_DIR, pflh.getDirectoryFileNames(DEM_RESOURCE_DIR)[-1]), "r") as f:
    dataDEM = f.read()

demData = np.zeros((180 * resolution + 1, 360 * resolution + 1))

# Extract elevation (DEM) data
rows = dataDEM.split("\n")[:181]
for latIdx, row in enumerate(rows):
    if row.strip():
        columns = row.split(",")[:-1]
        for longIdx, value in enumerate(columns):
            demData[latIdx, longIdx] = float(value)

binaryDem = np.where(demData <= 0, 0, 1)

plt.figure(figsize=(20, 10))
plt.imshow(binaryDem, cmap="viridis")
plt.colorbar(label="Elevation")
plt.title(f"Digital Elevation Model at {resPrefix}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)
plt.show()



# Initialise the new map (for todays grid) with default value 1 (unsuitable)
newMap = np.ones_like(projectionData, dtype=int)

# Map the projected coordinates to the DEM data
for i, triassicIndex in enumerate(projectionData):
    triassicIndex = int(triassicIndex)
    if 0 <= triassicIndex < binaryDem.flatten().size:
        newMap[i] = binaryDem.flatten()[triassicIndex]
    else:
        # Default to 0 if we are out of bounds
        newMap[i] = 1

# Load the ocean mask
oceanMask = Image.open(os.path.join(OCEANMASK_RESOURCE_DIR, pflh.getDirectoryFileNames(OCEANMASK_RESOURCE_DIR)[0])).convert("L")
oceanMaskArray = np.array(oceanMask)
oceanMaskBinary = (oceanMaskArray > 128).astype(int)

# Reshape the new map to the same shape as the DEM data
newMap = newMap.reshape((180 * resolution + 1, 360 * resolution + 1))
# Apply the ocean mask to the DEM map whilst conserving the unsuitability values
#newMap = np.minimum(newMap, oceanMaskBinary)

# Visualise the new map
plt.figure(figsize=(20, 10))
plt.imshow(newMap, cmap="viridis")
plt.colorbar(label="Unsuitability (1 = Unsuitable, 0 = Suitable)")
plt.title("Projected Unsuitability Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)
plt.show()

# Save the new map
outFile = (pfl.DATASET_DIR / f"{resPrefix}GROUNDTRUTHS") / f"groundTruth_resolutionScale_{resolution}.npy"
if not os.path.isfile(outFile):
    np.save(outFile, newMap)