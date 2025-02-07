import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from MAIN import GPLATES_ROTATION_MODEL as gprm
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This script generates all the permutations of longitudes and latitudes into an appropriate sized cell resolution, then projects
# them to the specified time interval. These projections are saved as index pointers, the final output is a 1D array of indices which
# point to the index of the cell at the specified time slice.
# This is useful for quickly accessing cell focused data as we can avoid the expensive projection calculations during runtime dataset generation.

# Specify the resolution, technically 'infinite' and therefore can be downscaled to any resolution
resolutionScale = 1
# Specify the steps between each time slice in MYA
timestep = 5
# How many time slices we are looking to project to (50 = 0 -> 245 MYA)
count = 50

# Create a directory for the Coordinate Projections
resPrefix = f"{1 / resolutionScale}x{1 / resolutionScale}"
COORDS_DIR = pfl.DATASET_DIR / f"{resPrefix}COORDS_PALEOMAPV19o_r1c"
pflh.createDirectoryIfNotExist(COORDS_DIR)

# Create a list of all combinations of longitudes and latitudes
longitudes, latitudes = np.meshgrid(np.arange(-180, 180 + (1 / resolutionScale), 1 / resolutionScale),
                                    np.arange(90, -90 - (1 / resolutionScale), -1 / resolutionScale))
# Create all permutations of longitudes and latitudes
coordPairs = np.column_stack((longitudes.ravel(), latitudes.ravel()))

# Calculate grid dimensions only once
longShape = 360 * resolutionScale + 1
latShape = 180 * resolutionScale + 1

# Iterate through the count times to produce the fossil projections
for t in tqdm(range(0, count), desc="Projecting Coordinates"):
    # Initialise the projection target
    projectionTarget = np.zeros((longShape * latShape))

    # Skip the projection if t = 0, can't project what is the current day
    if t > 0:
        # Batch project because that's far more efficient than single projection
        projectedLong, projectedLat = gprm.getPalaeoCoordinates(t * timestep, coordPairs[:, 0], coordPairs[:, 1])
    else:
        projectedLong, projectedLat = coordPairs[:, 0], coordPairs[:, 1]

    # Vectorised calculation of indices and ensure they are within bounds
    longIdx = np.clip(((projectedLong + 180) * resolutionScale).astype(int), 0, longShape - 1)
    latIdx = np.clip(((90 - projectedLat) * resolutionScale).astype(int), 0, latShape - 1)

    # Calculate the flattened indices (one dimensional array)
    projectionTarget = latIdx * longShape + longIdx

    print(projectionTarget[5000:5050])
    # Plot this to see what it looks like (kinda funky? Seems like it works)
    plt.figure(figsize=(20, 10))
    plt.imshow(projectionTarget.reshape((latShape, longShape)), cmap="terrain", interpolation="nearest")
    plt.title(f"Projection Target at Time Slice {t + 1}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Projected Index")
    plt.grid(False)
    plt.show()

    # Save it
    outFile = COORDS_DIR / f"{resPrefix}_projectionCoords_resolutionScale_{resolutionScale}_T{t + 1}.npy"
    if not os.path.isfile(outFile):
        np.save(outFile, projectionTarget)