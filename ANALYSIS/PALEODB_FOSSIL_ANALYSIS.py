import os
import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from MAIN import GPLATES_ROTATION_MODEL as gprm
from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# MAIN ------------------------------------------------------------------------

# This script is used to inspect the raw PBDB occurrence CSVs and do some preliminary cleaning and analysis of the query composition

# 1 = 1 degree, 181x361 = 65431, 10 = 0.1 degree, 1801x3601 = 6483601
resolution = 4
resPrefix = f"{1 / resolution}x{1 / resolution}"
FOSSIL_OCCURRENCE_DIR = pfl.DATASET_DIR / "FOSSIL_OCCURRENCE_ANALYSIS"

# Read the fossil occurrences from the files in the directory
for file in tqdm(pflh.getDirectoryFileNames(FOSSIL_OCCURRENCE_DIR), desc="Cleaning Fossil Data"):
    #file = pflh.getDirectoryFileNames(FOSSIL_OCCURRENCE_DIR)[0]
    with open(os.path.join(FOSSIL_OCCURRENCE_DIR, file), "r", encoding="utf-8") as f:
        data = f.read()

    # Extract relevant rows, skipping headers
    rows = data.split("\n")[21:-1]

    # Define the dictionary to store the fossil record data
    fossilData = {
        "occurrenceNo": [],
        "collectionNo": [],
        "longitude": [],
        "latitude": [],
        "minMa": [],
        "maxMa": [],
        "environment": [],
        "timeBins": []
    }

    # Process each row
    for i in tqdm(range(0, len(rows)), desc="Reading Fossil Data"):
        try:
            row = rows[i].split(",")

            # Extract relevant columns from the row if data exists
            occurrenceNo = int(row[0].replace('"', ""))
            collectionNo = int(row[4].replace('"', ""))
            long = float(row[17].replace('"', ""))
            lat = float(row[18].replace('"', ""))
            minMa = float(row[15].replace('"', ""))
            maxMa = float(row[14].replace('"', ""))
            environment = row[-2].replace('"', "").strip()
            timeBins = row[-1].replace('"', "").strip()

            # Ensure valid coordinate range
            if -180 <= long < 180 and -90 <= lat <= 90:
                # Ensure all relevant data is present
                if all([occurrenceNo, collectionNo, long, lat, minMa, maxMa, environment, timeBins]):
                    fossilData["occurrenceNo"].append(occurrenceNo)
                    fossilData["collectionNo"].append(collectionNo)
                    fossilData["longitude"].append(long)
                    fossilData["latitude"].append(lat)
                    fossilData["minMa"].append(minMa)
                    fossilData["maxMa"].append(maxMa)
                    fossilData["environment"].append(environment)
                    fossilData["timeBins"].append(timeBins)
        except:
            continue

    print(f"Unique Time Bins: {set(fossilData['timeBins'])}")
    df = pd.DataFrame(fossilData)

    # Calculate and display max and min ages
    print(f"Maximum Age (Ma): {df['maxMa'].max()}")
    print(f"Minimum Age (Ma): {df['minMa'].min()}")

    # Visualise the raw max and min ages
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df["maxMa"], bins=20, color="green", edgecolor="black")
    plt.title("Max Age Distribution")
    plt.xlabel("Max Age (Myr)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(df["minMa"], bins=20, color="cyan", edgecolor="black")
    plt.title("Min Age Distribution")
    plt.xlabel("Min Age (Myr)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Filter fossil records where the age is too far unreliable, an additional 2Myr buffer is ensured to maximise count
    df = df[(df["minMa"] >= int(file.split('_')[0]) - 2) & (df["maxMa"] <= int(file.split('_')[1]) + 2)]

    # Visualise the cleaned max and min plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df["maxMa"], bins=20, color="green", edgecolor="black")
    plt.title("Max Age Distribution")
    plt.xlabel("Max Age (Myr)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(df["minMa"], bins=20, color="cyan", edgecolor="black")
    plt.title("Min Age Distribution")
    plt.xlabel("Min Age (Myr)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Confirm values are within range
    print(f"New Maximum Age (Ma): {df['maxMa'].max()}")
    print(f"New Minimum Age (Ma): {df['minMa'].min()}")

    # Calculate the max and min ages per environment (Not really needed but a nice visualisation)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for env in df["environment"].unique():
        subset = df[df["environment"] == env]
        plt.hist(subset["maxMa"], bins=30, alpha=0.5, label=env)
    plt.title("Max Age Distribution Per Environment")
    plt.xlabel("Max Age (Myr)")
    plt.ylabel("Frequency")
    plt.legend(loc="right", fontsize=8)

    plt.subplot(1, 2, 2)
    for env in df["environment"].unique():
        subset = df[df["environment"] == env]
        plt.hist(subset["minMa"], bins=30, alpha=0.5, label=env)
    plt.title("Min Age Distribution Per Environment")
    plt.xlabel("Min Age (Myr)")
    plt.ylabel("Frequency")
    #plt.tight_layout()
    plt.show()

    # Display summary
    print(f"Total Fossil Records Processed: {len(df)}")
    print(df.head())

    # Plot the distribution of fossil occurrences across time bins
    plt.figure(figsize=(12, 9))
    plt.bar(df["timeBins"].value_counts().index, df["timeBins"].value_counts().values)
    plt.xlabel("Time Bins")
    plt.ylabel("Count")
    plt.title("Distribution of Fossil Occurrences Across Time Bins")
    plt.xticks(rotation=90, ha="right")
    plt.show()

    # Export the DataFrame to a CSV file
    pflh.createDirectoryIfNotExist(pathlib.Path(FOSSIL_OCCURRENCE_DIR) / "CLEANED")
    exportPath = pathlib.Path(FOSSIL_OCCURRENCE_DIR) / "CLEANED" / f"{file.split('.')[0]}_Cleaned.csv"
    # Remove the unknown time binned fossils as they are currently unreliable
    df.to_csv(exportPath, index=False, encoding="utf-8")
    print(f"Cleaned Data exported!")

    # Print some stats based on the cleaning
    print(f"---------------------")
    print(f"Total Fossil Records Pre-Cleaning: {len(rows)}")
    print(f"Total Fossil Records Post-Cleaning: {len(df)}")
    print("Percentage Removed: {:.2f}%".format((1 - len(df) / len(rows)) * 100))
    print(f"---------------------")
