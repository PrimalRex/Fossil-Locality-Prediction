import os

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This script converts any exported GIS CSVs from CSV format to NumPy format ready for any models to process.

# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 4
# Define the path to resources given by our input
resPrefix = f"{1 / resolution}x{1 / resolution}"
# Folder name without prefix
suffix = "NDVI"

# Our main resource
CSV_DIR = pfl.RESOURCES_DIR / f"{resPrefix}{suffix}"
# Create a directory for our stored NumPy files
NPY_DIR = pfl.DATASET_DIR / f"{resPrefix}{suffix}"
pflh.createDirectoryIfNotExist(NPY_DIR)

# Whether we want to compress the files or not
# Uncompress ~25MB, Compress ~1-5MB
useCompression = False

for i, file in enumerate(
        tqdm(pflh.getDirectoryFileNames(CSV_DIR), desc="Converting CSVs to NumPy")):
            name = file.split(".")[0]
            if not os.path.isfile(NPY_DIR / f"{name}.npy"):
                with open(os.path.join(CSV_DIR, file), 'r') as f:
                    data = f.read()
                    arrayData = np.zeros((180 * resolution + 1) * (360 * resolution + 1))
                    rows = data.split("\n")[1:-1]
                    for j in range(len(rows)):
                        arrayData[j] = float(rows[j].split(",")[0])

                if useCompression:
                    np.savez_compressed(NPY_DIR / f"{name}.npz", arrayData)
                else:
                    np.save(NPY_DIR / f"{name}.npy", arrayData)
