import os

from MAIN import PFL_PATHS as pfl
import numpy as np
from matplotlib import pyplot as plt

with open(os.path.join(pfl.MODELS_OUTPUT_DIR, "globalPredictions_LSTMss.csv"), "r") as f:
    predictions = f.read()

predictions = predictions.split("\n")[1:-1]
predictionNMPY = np.zeros((181 * 361))
knownFossilsNMPY = np.zeros((181 * 361))
for i in range(0, len(predictions)):
    predictionNMPY[i] = float(predictions[i].split(",")[0])
    knownFossilsNMPY[i] = float(predictions[i].split(",")[1])

# Plot the prediction weightings and then overlay the fossils that the model for this was trained on
plt.figure(figsize=(20, 10))
sc = plt.imshow(predictionNMPY.reshape(181, 361), cmap="viridis", interpolation="nearest")
#plt.colorbar(sc)
sc = plt.imshow(knownFossilsNMPY.reshape(181, 361), cmap="gray", interpolation="nearest", alpha=.4)
plt.title("Predictions")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)
plt.show()

