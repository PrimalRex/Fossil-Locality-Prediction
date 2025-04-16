import numpy as np
from sklearn.metrics import recall_score

# MAIN ------------------------------------------------------------------------

# Measures the IoU, aka Spatial Accuracy in relation to the ground truth and predictions
def IoUScore (groundTruths, predictions):
    # Find the intersection by summing the cells where both ground truth and predictions are 1
    intersection = np.sum(np.where((groundTruths == 1) & (predictions == 1), 1, 0))
    # Find the union by summing the cells where either ground truth or predictions are 1
    union = np.sum(np.where((groundTruths == 1) | (predictions == 1), 1, 0))

    # Compute IoU
    if(union == 0):
        # Safe divide by 0
        return 0.0

    return float(intersection) / float(union)