import os

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

import PFL_HELPER as pflh
import PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# Our main resource are the netCDF .1x.1 DEMs
POINT_ONE_DEG_NC_DEMS_DIR = pfl.RESOURCES_DIR / "0.1x0.1DEMs_NC"
# Create a directory for our stored NumPy files
POINT_ONE_DEG_DEMS_DIR = pfl.RESOURCES_DIR / "0.1x0.1DEMs"
pflh.createDirectoryIfNotExist(POINT_ONE_DEG_DEMS_DIR)

# Whether we want to compress the files or not
# Uncompress ~25MB, Compress ~1-5MB
useCompression = True

for i, file in enumerate(
        tqdm(pflh.getDirectoryFileNames(POINT_ONE_DEG_NC_DEMS_DIR), desc="Converting .1x.1 DEMs to NumPy")):
            name = file.split(".")[0]
            if not os.path.isfile(POINT_ONE_DEG_DEMS_DIR / f"{name}_3601.npy"):
                f = Dataset(os.path.join(POINT_ONE_DEG_NC_DEMS_DIR, file), "r")
                elevation = np.array(f.variables[list(f.variables.keys())[2]][:])

                if useCompression:
                    np.savez_compressed(POINT_ONE_DEG_DEMS_DIR / f"{name}_3601.npz", elevation)
                else:
                    np.save(POINT_ONE_DEG_DEMS_DIR / f"{name}_3601.npy", elevation)
