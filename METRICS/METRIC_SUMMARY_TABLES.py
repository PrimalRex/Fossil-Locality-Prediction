import numpy as np
from sklearn.metrics import recall_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from METRICS.METRIC_FOSSIL_CONFIDENCE import fossiliferousConfidenceScore

# MAIN ------------------------------------------------------------------------

# Produces a list of metrics results based on a test set
# Additionally produces a random guessing baseline for comparison
def displayMetricsAgainstRandomGuessing(yTrain, yTest, testPredictions, testBinaryPredictions, prefix="Test"):
    print(f"----------------------------")
    print(f"{prefix} Overall Accuracy: {accuracy_score(yTest, testBinaryPredictions):.4f}")
    print(f"{prefix} Precision (Sedimentary): {precision_score(yTest, testBinaryPredictions, zero_division=0):.4f}")
    print(f"{prefix} Recall (Sedimentary): {recall_score(yTest, testBinaryPredictions, zero_division=0):.4f}")
    print(f"{prefix} Fossiliferous Confidence (Sedimentary): {fossiliferousConfidenceScore(yTest, testPredictions):.4f}")
    print(f"{prefix} AUC-ROC: {roc_auc_score(yTest, testPredictions):.4f}")

    print(f"----------------------------")

    randomTestPredictions = np.random.choice([0, 1], len(yTest), True, [1 - np.mean(yTrain), np.mean(yTrain)])
    print(f"{prefix} Random Guessing Theoretical Baseline: {len(yTest[yTest == 1]) / len(yTest):.4f}")
    print(f"{prefix} Random Baseline Accuracy: {accuracy_score(yTest, randomTestPredictions):.4f}")
    print(f"{prefix} Random Baseline Precision (Sedimentary): {precision_score(yTest, randomTestPredictions, zero_division=0):.4f}")
    print(f"{prefix} Random Baseline Recall (Sedimentary): {recall_score(yTest, randomTestPredictions, zero_division=0):.4f}")
    print(f"{prefix} Random Baseline Fossiliferous Confidence (Sedimentary): {fossiliferousConfidenceScore(yTest, randomTestPredictions):.4f}")
    print(f"{prefix} AUC-ROC: {roc_auc_score(yTest, randomTestPredictions):.4f}")
    print(f"----------------------------")