import numpy as np


def CalculateAllMetricsUpdated(cm, epsilon=1e-7):
  """
  Calculate various classification metrics from a confusion matrix using Macro, Micro, and Weighted averaging.

  This function computes a comprehensive set of classification metrics based on the provided confusion matrix.
  Metrics are calculated using three averaging methods: Macro, Micro, and Weighted. Additionally, the function
  ensures that all calculations are numerically stable by avoiding division by zero using a small epsilon value.

  Parameters:
  ----------
  cm : numpy.ndarray
      The confusion matrix of shape (noClasses, noClasses), where `noClasses` is the number of unique classes.
      Each element `cm[i, j]` represents the count of instances classified as class `j`
      but actually belonging to class `i`.

  epsilon : float, optional
      A small value added to denominators to prevent division by zero. Default is 1e-7.

  Returns:
  -------
  results : dict
      A dictionary containing the following metrics for each averaging method (Macro, Micro, Weighted):
          - Precision: The ratio of correctly predicted positive observations to total predicted positives.
          - Recall: The ratio of correctly predicted positive observations to all actual positives.
          - F1 Score: The harmonic mean of Precision and Recall.
          - Accuracy: The ratio of correctly predicted observations to total observations.
          - Specificity: The ratio of correctly predicted negatives to total actual negatives.
          - Balanced Accuracy (BAC): The average of Recall and Specificity.
          - Fowlkes-Mallows Index: The geometric mean of Precision and Recall.
          - Geometric Mean (GM): The square root of the product of Precision and Recall.
          - Average of all metrics (AVG): The mean of all computed metrics.
          - SCORE: The final weighted average score, constrained between 0 and 1.

  Notes:
  -----
  - Macro averaging calculates metrics for each class independently and then takes the unweighted average.
  - Micro averaging aggregates contributions from all classes before computing metrics.
  - Weighted averaging computes metrics for each class and then takes the weighted average based on class support.
  - The final SCORE is derived as the weighted average of all metrics and is ensured to be within [0, 1].
  """

  # Handle binary classification case.
  if (cm.shape[0] <= 2):  # Binary classification.
    tp, fp, fn, tn = cm.ravel()  # Extract scalar values for True Positives, False Positives, etc.
  else:  # Multi-class classification.
    tp = np.diag(cm)  # Diagonal elements represent True Positives for each class.
    fp = np.sum(cm, axis=0) - tp  # Column sums minus True Positives give False Positives.
    fn = np.sum(cm, axis=1) - tp  # Row sums minus True Positives give False Negatives.
    tn = np.sum(cm) - (tp + fp + fn)  # Remaining elements represent True Negatives.

  results = {}  # Initialize dictionary to store all calculated metrics.

  # Store raw counts of TP, FP, FN, and TN.
  results.update({
    "TP": tp,
    "FP": fp,
    "FN": fn,
    "TN": tn,
  })

  # ======================
  # Macro Averaging
  # ======================

  # Calculate macro precision.
  precisionMacro = np.mean(tp / (tp + fp + epsilon))

  # Calculate macro recall.
  recallMacro = np.mean(tp / (tp + fn + epsilon))

  # Calculate macro F1 score.
  f1Macro = 2 * precisionMacro * recallMacro / (precisionMacro + recallMacro + epsilon)

  # Calculate macro accuracy.
  accuracyMacro = np.mean((tp + tn) / (np.sum(cm) + epsilon))

  # Calculate macro specificity.
  specificityMacro = np.mean(tn / (tn + fp + epsilon))

  # Calculate macro balanced accuracy.
  bacMacro = (recallMacro + specificityMacro) / 2.0

  # Calculate macro Fowlkes-Mallows index.
  fowlkesMacro = np.sqrt(precisionMacro * recallMacro)

  # Calculate macro geometric mean.
  gmMacro = np.sqrt(precisionMacro * recallMacro)

  # Calculate macro average of all metrics.
  avgMacro = (
               precisionMacro + recallMacro + f1Macro + accuracyMacro + specificityMacro +
               bacMacro + fowlkesMacro + gmMacro
             ) / 8.0

  # Store macro-averaged metrics.
  results.update({
    "Macro Precision"  : precisionMacro,
    "Macro Recall"     : recallMacro,
    "Macro F1"         : f1Macro,
    "Macro Accuracy"   : accuracyMacro,
    "Macro Specificity": specificityMacro,
    "Macro BAC"        : bacMacro,
    "Macro Fowlkes"    : fowlkesMacro,
    "Macro GM"         : gmMacro,
    "Macro AVG"        : avgMacro,
  })

  # ======================
  # Micro Averaging
  # ======================

  # Calculate micro precision.
  precisionMicro = np.sum(tp) / (np.sum(tp + fp) + epsilon)

  # Calculate micro recall.
  recallMicro = np.sum(tp) / (np.sum(tp + fn) + epsilon)

  # Calculate micro F1 score.
  f1Micro = 2 * precisionMicro * recallMicro / (precisionMicro + recallMicro + epsilon)

  # Calculate micro accuracy.
  accuracyMicro = np.sum(tp + tn) / (np.sum(tp + tn + fp + fn) + epsilon)

  # Calculate micro specificity.
  specificityMicro = np.sum(tn) / (np.sum(tn + fp) + epsilon)

  # Calculate micro balanced accuracy.
  bacMicro = (recallMicro + specificityMicro) / 2.0

  # Calculate micro Fowlkes-Mallows index.
  fowlkesMicro = np.sqrt(precisionMicro * recallMicro)

  # Calculate micro geometric mean.
  gmMicro = np.sqrt(precisionMicro * recallMicro)

  # Calculate micro average of all metrics.
  avgMicro = (
               precisionMicro + recallMicro + f1Micro + accuracyMicro + specificityMicro +
               bacMicro + fowlkesMicro + gmMicro
             ) / 8.0

  # Store micro-averaged metrics.
  results.update({
    "Micro Precision"  : precisionMicro,
    "Micro Recall"     : recallMicro,
    "Micro F1"         : f1Micro,
    "Micro Accuracy"   : accuracyMicro,
    "Micro Specificity": specificityMicro,
    "Micro BAC"        : bacMicro,
    "Micro Fowlkes"    : fowlkesMicro,
    "Micro GM"         : gmMicro,
    "Micro AVG"        : avgMicro,
  })

  # ======================
  # Weighted Averaging
  # ======================

  # Calculate number of samples per class.
  samples = np.sum(cm, axis=1)

  # Calculate class weights based on sample size.
  weights = samples / np.sum(cm)

  # Calculate weighted precision.
  precisionWeighted = np.sum((tp / (tp + fp + epsilon)) * weights)

  # Calculate weighted recall.
  recallWeighted = np.sum((tp / (tp + fn + epsilon)) * weights)

  # Calculate weighted F1 score.
  f1Weighted = 2 * precisionWeighted * recallWeighted / (precisionWeighted + recallWeighted + epsilon)

  # Calculate weighted accuracy.
  accuracyWeighted = np.sum((tp + tn) * weights) / np.sum(cm)

  # Calculate weighted specificity.
  specificityWeighted = np.sum((tn / (tn + fp + epsilon)) * weights)

  # Calculate weighted balanced accuracy.
  bacWeighted = (recallWeighted + specificityWeighted) / 2.0

  # Calculate weighted Fowlkes-Mallows index.
  fowlkesWeighted = np.sqrt(precisionWeighted * recallWeighted)

  # Calculate weighted geometric mean.
  gmWeighted = np.sqrt(precisionWeighted * recallWeighted)

  # Calculate weighted average of all metrics.
  avgWeighted = (
                  precisionWeighted + recallWeighted + f1Weighted + accuracyWeighted + specificityWeighted +
                  bacWeighted + fowlkesWeighted + gmWeighted
                ) / 8.0

  # Store weighted-averaged metrics.
  results.update({
    "Weighted Precision"  : precisionWeighted,
    "Weighted Recall"     : recallWeighted,
    "Weighted F1"         : f1Weighted,
    "Weighted Accuracy"   : accuracyWeighted,
    "Weighted Specificity": specificityWeighted,
    "Weighted BAC"        : bacWeighted,
    "Weighted Fowlkes"    : fowlkesWeighted,
    "Weighted GM"         : gmWeighted,
    "Weighted AVG"        : avgWeighted,
  })

  # Final SCORE calculation.
  results["SCORE"] = avgWeighted  # Use weighted average as the final SCORE.

  # Ensure SCORE is within [0, 1].
  if (results["SCORE"] < 0.0):
    results["SCORE"] = 0.0  # Clamp to zero if negative.

  if (results["SCORE"] > 1.0):
    results["SCORE"] = 1.0  # Clamp to one if greater than one.

  if (np.isnan(results["SCORE"]) or np.isinf(results["SCORE"])):
    results["SCORE"] = 0.0  # Set to zero if NaN or infinite.

  return results  # Return the dictionary containing all calculated metrics.


if __name__ == "__main__":
  # Define a confusion matrix for a multi-class classification problem.
  cm = np.array([
    [13, 2, 0],  # Class 0 predictions.
    [3, 15, 0],  # Class 1 predictions.
    [0, 0, 10],  # Class 2 predictions.
  ])

  # Calculate all metrics using the confusion matrix.
  results = calculateAllMetricsUpdated(cm)

  # Print the calculated metrics.
  for key, value in results.items():
    print(f"{key}: {np.round(value, 4)}")

  # Example output:
  # TP: [13 15 10]
  # FP: [3 2 0]
  # FN: [2 3 0]
  # TN: [25 23 33]
  # Macro Precision: 0.8983
  # Macro Recall: 0.9
  # Macro F1: 0.8991
  # Macro Accuracy: 0.9225
  # Macro Specificity: 0.9376
  # Macro BAC: 0.9188
  # Macro Fowlkes: 0.8991
  # Macro GM: 0.8991
  # Macro AVG: 0.9093
  # Micro Precision: 0.8837
  # Micro Recall: 0.8837
  # Micro F1: 0.8837
  # Micro Accuracy: 0.9225
  # Micro Specificity: 0.9419
  # Micro BAC: 0.9128
  # Micro Fowlkes: 0.8837
  # Micro GM: 0.8837
  # Micro AVG: 0.8995
  # Weighted Precision: 0.8853
  # Weighted Recall: 0.8837
  # Weighted F1: 0.8845
  # Weighted Accuracy: 0.9108
  # Weighted Specificity: 0.9291
  # Weighted BAC: 0.9064
  # Weighted Fowlkes: 0.8845
  # Weighted GM: 0.8845
  # Weighted AVG: 0.8961
  # SCORE: 0.8961
