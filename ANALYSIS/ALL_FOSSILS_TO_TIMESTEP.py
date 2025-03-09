import os
import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from MAIN import GPLATES_ROTATION_MODEL as gprm
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# FUNCTION ---------------------------------------------------------------------

# This function consumes the defined binlist and an age, it then finds the nearest bin to the age and returns it
def findNearestBin(age):
    # Assume bin 0 is the closest
    closestBin = bins[0]
    currentDiff = abs(age - closestBin)

    # Loop through all the bins to find the bin with the smallest age difference
    for bin in bins:
        difference = abs(age - bin)
        if difference < currentDiff:
            closestBin = bin
            currentDiff = difference

    # Return the closest bin
    return closestBin

# MAIN ------------------------------------------------------------------------

# This tool allows us to take our cleaned fossil occurrences and bin them by a timestep interval to be used in other tools where we may need to look at occurrences per timestep.

# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 4
binStep = 5
resPrefix = f"{1 / resolution}x{1 / resolution}"

# Define binStep list (0, 5, 10, ..., 245)
# Since our dataset never involves data from 0-25, we don't need to bin anything to them but will keep them for consistency with other tools
bins = list(range(0, 555, binStep))

# Temporary list to hold all fossil data
allFossils = []

CLEANED_FOSSIL_OCCURRENCE_DIR = pfl.DATASET_DIR / "FOSSIL_OCCURRENCE_ANALYSIS" / "CLEANED"
BINNED_FOSSILS_DIR = pfl.DATASET_DIR / "FOSSIL_OCCURRENCE_ANALYSIS" / f"BINNED_BY_{binStep}Myr"
pflh.createDirectoryIfNotExist(BINNED_FOSSILS_DIR)

# Read the fossil occurrences from the directory
for file in tqdm(pflh.getDirectoryFileNames(CLEANED_FOSSIL_OCCURRENCE_DIR), desc="Binning Fossil Data"):
    with open(os.path.join(CLEANED_FOSSIL_OCCURRENCE_DIR, file), "r", encoding="utf-8") as f:
        df = pd.read_csv(f)
        allFossils.append(df)
df = pd.concat(allFossils, ignore_index=True)

# Create a column to find an approximate age
# NOTE: Classifications may benefit from a different heuristic approximation of ages, for the most even an average is used
df["midAge"] = (df["minMa"] + df["maxMa"]) / 2
# Find the bin based on that age
df["binnedAge"] = df["midAge"].apply(findNearestBin)

for bin in bins:
    # Select all the fossils that fall within each bin and save them
    binDF = df[df["binnedAge"] == bin]
    binDF.to_csv(os.path.join(BINNED_FOSSILS_DIR, f"{bin:03d}.csv"), index=False)
    print(f"Saved {len(binDF)} fossils to {bin}.csv")