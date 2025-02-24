import numpy as np
from sklearn.metrics import recall_score

# MAIN ------------------------------------------------------------------------

# Measures the confidence in the model through its prediction in each cell. This metric focuses
# on finding trustworthy confidence levels to show evidence for a correct prediction with unknown ground truth cells.
def fossiliferousConfidenceScore (groundTruths, predictions):
    # Convert from softmax to binary if necessary
    if len(groundTruths.shape) > 1 and groundTruths.shape[1] > 1:
        groundTruths = np.argmax(groundTruths, axis=1)
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Ensure both arrays are float type
    predictions = np.array(predictions).astype(float)
    groundTruths = np.array(groundTruths).astype(float)

    # Total presence score (predicted confidence * ground truth)
    totalPresenceScore = np.sum(predictions * groundTruths)

    # Total predicted score (sum of predicted confidence scores)
    totalPredictedScore = np.sum(predictions)

    # Check if our predicted score is 0, if so don't need to do any calculation as div 0 will be 0
    if totalPredictedScore == 0:
        return 0.0

    return float(totalPresenceScore) / float(totalPredictedScore)


# EXAMPLE USAGE ----------------------------------------------------------------

# # Say we produce these predicted scores against these ground truths
# predictedScores = np.array([0.5, 0.1, 0.7, 0.8, 0.91])
# groundTruth = np.array([1, 0, 1, 1, 0])
#
# # The confidence doesn't need any binary conversion of the predicted scores, we can guess based on the confidence of each cell
# # rather than assigning a threshold to convert to binary. Effectively, the prediction would be able to score 1.0 Recall but a low confidence if the confidence score is significantly < 1.0
# confidenceMetric = fossiliferousConfidenceScore(groundTruth, predictedScores)
# print(f"Fossiliferous Confidence Metric: {confidenceMetric:.4f}")
#
# # The recall returns a 1.0, aka all true positives predicted, this does not factor any confidence, therefore we get a weak result dependent on threshold.
# binaryPredictions = (predictedScores >= 0.5).astype(int)
# recallScore = recall_score(groundTruth, binaryPredictions, zero_division=0)
# print(f"Recall Score: {recallScore:.4f}")