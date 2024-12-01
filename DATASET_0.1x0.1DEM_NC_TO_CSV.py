import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from netCDF4 import Dataset
from tqdm import tqdm

import PFL_HELPER as pflh
import PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# WARNING: This script will extract and populate the .1x.1 DEMs as CSV files completely uncompressed.
# This will take approx 200MB per DEM file. Ensure you have enough disk space if you wish to proceed.

# Our main resource are the netCDF .1x.1 DEMs
POINT_ONE_DEG_NC_DEMS_DIR = pfl.RESOURCES_DIR / "0.1x0.1DEMs_NC"
# Create a directory for our stored CVS files
POINT_ONE_DEG_DEMS_DIR = pfl.RESOURCES_DIR / "0.1x0.1DEMs"
pflh.createDirectoryIfNotExist(POINT_ONE_DEG_DEMS_DIR)

for i, file in enumerate(
        tqdm(pflh.getDirectoryFileNames(POINT_ONE_DEG_NC_DEMS_DIR), desc="Converting .1x.1 DEMs to CSV")):
        name = file.split(".")[0]
        if not os.path.isfile(POINT_ONE_DEG_DEMS_DIR / f"{name}_3601.csv"):
            f = Dataset(os.path.join(POINT_ONE_DEG_NC_DEMS_DIR, file), "r")

            longitude = f.variables["longitude"][:]
            latitude = f.variables["latitude"][:]
            elevation = f.variables[list(f.variables.keys())[2]][:]

            plt.figure(figsize=(20, 10))
            plt.imshow(elevation, cmap="viridis")
            # plt.colorbar(label="Elevation (or other variable)")
            plt.title("Spatial Data Visualization")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.show()

            # Ensure that it follows the existing 1x1 CSV format
            longitudes, latitudes = np.meshgrid(f.variables["longitude"][:], f.variables["latitude"][:])
            df = pd.DataFrame(
                {"longitude": longitudes.ravel(), "latitude": latitudes.ravel(), "elevation": elevation.ravel()})
            df.to_csv(POINT_ONE_DEG_DEMS_DIR / f"{name}_3601.csv", index=False)
