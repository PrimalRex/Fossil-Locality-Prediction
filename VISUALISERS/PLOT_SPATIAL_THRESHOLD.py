import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

from METRICS.METRIC_IOU import IoUScore


# MAIN ------------------------------------------------------------------------

# This function produces a plot of metrics at different thresholds to find the optimal threshold
def displayThresholdImpact(groundTruths, predictions, min, max, step=0.01):
    thresholds = np.arange(min, max, step)
    precisions = []
    recalls = []
    ious = []

    # Compute metrics for each threshold
    for threshold in thresholds:
        binaryPredictions = (predictions >= threshold).astype(int)
        precision = precision_score(groundTruths, binaryPredictions, zero_division=0)
        recall = recall_score(groundTruths, binaryPredictions, zero_division=0)
        iou = IoUScore(groundTruths, binaryPredictions)

        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, recalls, label="Recall", marker="o")
    plt.plot(thresholds, precisions, label="Precision", marker="o")
    plt.plot(thresholds, ious, label="IoU", marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title("Threshold Impact")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
