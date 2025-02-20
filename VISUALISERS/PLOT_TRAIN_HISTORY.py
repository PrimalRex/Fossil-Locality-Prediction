import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# MAIN ------------------------------------------------------------------------

# This script is for training history related visualisations

# Plots a smoothened graph of the training and validation accuracy
def plotAccuracy(history):
    trainAcc = history.history["accuracy"]
    valAcc = history.history["val_accuracy"]
    epochs = np.arange(len(trainAcc))

    # Smooth the curves
    epochsScaled = np.linspace(epochs.min(), epochs.max(), 300)
    trainSpline = make_interp_spline(epochs, trainAcc, k=3)
    valSpline = make_interp_spline(epochs, valAcc, k=3)

    plt.figure(figsize=(10, 5))
    plt.plot(epochsScaled, trainSpline(epochsScaled), linestyle='-', label="Training Accuracy")
    plt.plot(epochsScaled, valSpline(epochsScaled), linestyle='-', label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# Plots a smoothened graph of the training and validation loss
def plotLoss(history):
    trainLoss = history.history["loss"]
    valLoss = history.history["val_loss"]
    epochs = np.arange(len(trainLoss))

    # Smooth the curves
    epochsScaled = np.linspace(epochs.min(), epochs.max(), 300)
    trainSpline = make_interp_spline(epochs, trainLoss, k=3)
    valSpline = make_interp_spline(epochs, valLoss, k=3)

    plt.figure(figsize=(10, 5))
    plt.plot(epochsScaled, trainSpline(epochsScaled), linestyle='-', label="Training Loss")
    plt.plot(epochsScaled, valSpline(epochsScaled), linestyle='-', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()