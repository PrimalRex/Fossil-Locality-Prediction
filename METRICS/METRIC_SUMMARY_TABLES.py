import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from METRICS.METRIC_FOSSIL_CONFIDENCE import fossiliferousConfidenceScore

# MAIN ------------------------------------------------------------------------

# Produces a list of metrics results based on a test set
# Additionally produces a random guessing baseline for comparison
def displayMetricsAgainstRandomGuessing(yTrain, yTest, testPredictions, testBinaryPredictions, prefix="Test", showRandomTests=True):
    print(f"----------------------------")
    print(f"{prefix} Overall Accuracy: {accuracy_score(yTest, testBinaryPredictions):.4f}")
    print(f"{prefix} Precision: {precision_score(yTest, testBinaryPredictions, zero_division=0):.4f}")
    print(f"{prefix} Recall: {recall_score(yTest, testBinaryPredictions, zero_division=0):.4f}")
    print(f"{prefix} F1: {f1_score(yTest, testBinaryPredictions, zero_division=0):.4f}")
    print(f"{prefix} Fossiliferous Confidence: {fossiliferousConfidenceScore(yTest, testPredictions):.4f}")
    print(f"{prefix} AUC-ROC: {roc_auc_score(yTest, testPredictions):.4f}")

    print(f"----------------------------")

    if showRandomTests:
        randomTestPredictions = np.random.choice([0, 1], len(yTest), True, [1 - np.mean(yTrain), np.mean(yTrain)])
        print(f"{prefix} Random Guessing Theoretical Baseline: {len(yTest[yTest == 1]) / len(yTest):.4f}")
        print(f"{prefix} Random Baseline Accuracy: {accuracy_score(yTest, randomTestPredictions):.4f}")
        print(f"{prefix} Random Baseline Precision: {precision_score(yTest, randomTestPredictions, zero_division=0):.4f}")
        print(f"{prefix} Random Baseline Recall: {recall_score(yTest, randomTestPredictions, zero_division=0):.4f}")
        print(f"{prefix} Random Baseline F1: {f1_score(yTest, randomTestPredictions, zero_division=0):.4f}")
        print(f"{prefix} Random Baseline Fossiliferous Confidence: {fossiliferousConfidenceScore(yTest, randomTestPredictions):.4f}")
        print(f"----------------------------")

# Multi-class version of the above function
def displayMetricsAgainstRandomGuessingMultiClass(yTrain, yTest, testPredictions, testClassPredictions, prefix="Test", showRandomTests=True):
    print(f"----------------------------")
    print(f"{prefix} Overall Accuracy: {accuracy_score(yTest, testClassPredictions):.4f}")
    print(f"{prefix} Precision: {precision_score(yTest, testClassPredictions, average='macro', zero_division=0):.4f}")
    print(f"{prefix} Recall: {recall_score(yTest, testClassPredictions, average='macro', zero_division=0):.4f}")
    print(f"{prefix} F1: {f1_score(yTest, testClassPredictions, average='macro', zero_division=0):.4f}")
    #print(f"{prefix} Fossiliferous Confidence: {fossiliferousConfidenceScore(yTest, testClassPredictions):.4f}")
    print(f"{prefix} AUC-ROC: {roc_auc_score(yTest, testPredictions, multi_class='ovr', average='macro'):.4f}")

    print(f"----------------------------")

    if showRandomTests:
        randomTestPredictions = np.random.choice(np.unique(yTrain), len(yTest), True)
        print(f"{prefix} Random Guessing Theoretical Baseline: {1 / len(np.unique(yTest)):.4f}")
        print(f"{prefix} Random Baseline Accuracy: {accuracy_score(yTest, randomTestPredictions):.4f}")
        print(f"{prefix} Random Baseline Precision: {precision_score(yTest, randomTestPredictions, average='macro', zero_division=0):.4f}")
        print(f"{prefix} Random Baseline Recall: {recall_score(yTest, randomTestPredictions, average='macro', zero_division=0):.4f}")
        print(f"{prefix} Random Baseline F1: {f1_score(yTest, randomTestPredictions, average='macro', zero_division=0):.4f}")
        #print(f"{prefix} Random Baseline Fossiliferous Confidence: {fossiliferousConfidenceScore(yTest, randomTestPredictions):.4f}")
        print(f"----------------------------")