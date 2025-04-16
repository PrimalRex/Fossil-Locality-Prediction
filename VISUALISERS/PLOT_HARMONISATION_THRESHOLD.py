import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from METRICS.METRIC_IOU import IoUScore


# MAIN ------------------------------------------------------------------------

# This function produces a plot of metrics at different thresholds to find the optimal threshold
def displayHarmonisedThresholdImpact(groundTruths, predictions, predictionImprover, min, max, step=0.05):
    thresholds = np.arange(min, max, step)
    precisions = []
    recalls = []
    ious = []

    # Compute metrics for each threshold
    for threshold in thresholds:
        adjustedPredictions = np.copy(predictions)
        # For every prediction, increase the prediction score based on the improver's prediction
        adjustedPredictions[predictionImprover == 1] += threshold
        # Ensure the scores don't exceed a max of 1
        adjustedPredictions = np.clip(adjustedPredictions, 0, 1)
        adjustedBinaryPredictions = (adjustedPredictions >= 0.90).astype(int)

        precision = precision_score(groundTruths, adjustedBinaryPredictions, zero_division=0)
        recall = recall_score(groundTruths, adjustedBinaryPredictions, zero_division=0)
        iou = IoUScore(groundTruths, adjustedBinaryPredictions)
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
    plt.title("Harmonised Threshold Impact")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
