import os
import requests
from tqdm import tqdm

from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# Simple tool to download fossil occurrences by time step from the PaleoDB API
# Does not currently batch execute but provides the same desired effect to generate fossil data for multiple time steps
# TODO: Rework parameters into a function based system so this can be more of a 'manager' rather than a dependent script for 1 file

# Define the time range for intervals (Inclusive)
startMYA = 245
# End of range (Inclusive to the step size)
endMYA = 0
# Step size for intervals
step = 5
# How much to buffer the time range by (adds and subtracts from the current time step)
timeBuffer = 2.75
# Define the geological context parameters for the desired data
lithologies = "&lithology=!metamorphic,volcanic,unknown"
environments = "&envtype=terr,lacust,fluvial,karst,terrother"

# Create a directory for the fossil occurrences by time step
FOSSIL_TIMESTEP_DIR = pfl.DATASET_DIR / "FOSSIL" / f"{startMYA}_{endMYA}_FOSSIL_OCCURENCES_BY_TIMESTEP"
pflh.createDirectoryIfNotExist(FOSSIL_TIMESTEP_DIR)

# Iterate over the intervals and download data
for time in tqdm(range(startMYA, endMYA - step, -step), desc="Downloading Occurrences by Time Step"):
    url = (f"https://paleobiodb.org/data1.2/occs/list.csv?datainfo&rowcount&max_ma={time + timeBuffer}&min_ma={max(time - timeBuffer, 0)}"
           + lithologies + environments + f"&show=coords")
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        filename = os.path.join(FOSSIL_TIMESTEP_DIR, f"{time:03}_fossilDeathApproximations.csv")
        # Save the data to a CSV file
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Data for {time} MYA successfully saved to {filename}.")
    else:
        # Ensure that parameters are correct otherwise this error will throw
        print(f"Failed to download data for {time} MYA. Status code: {response.status_code}")
print("Download finished!")
