# Import necessary libraries.
import IgnoreWarnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def StoreArgumentsToText(storageDir, basePathNoExt, argsDict):
  """
  Store the command-line arguments in a text file for reference.

  Parameters:
  ----------
  storageDir : str
      Directory to store the text file.
  basePathNoExt : str
      Base name of the dataset file (without extension).
  argsDict : dict
      Dictionary containing all the arguments and their values.

  This function writes the arguments to a text file in the specified directory.
  """
  # Store the arguments in a txt file.
  with open(os.path.join(storageDir, f"Arguments.txt"), "w") as f:
    f.write("Arguments:\n")
    for key, value in argsDict.items():
      f.write(f"{key}: {value}\n")


def CalculateElapsedTime(startTime, endTime, argsDict, keyword, doStore=True, doVerbose=True):
  """
  Calculate and optionally store or print the elapsed time for the optimization process.

  Parameters:
  ----------
  startTime : float
      Start time of the optimization process (in seconds).
  endTime : float
      End time of the optimization process (in seconds).
  argsDict : dict
      Dictionary containing relevant parameters such as population size, number of iterations, etc.
  doStore : bool, optional
      Whether to store the elapsed time in a text file (default: True).
  doVerbose : bool, optional
      Whether to print the elapsed time to the console (default: True).

  Returns:
  -------
  dict
      A dictionary containing the total elapsed time and per-item elapsed time in hours, minutes, and seconds.
  """
  popSize = argsDict["popSize"]
  numIter = argsDict["numIter"]
  storageDir = argsDict["storageDir"]

  # Calculate the total elapsed time.
  deltaTime = endTime - startTime
  deltaTimeSec = deltaTime % 60
  deltaTimeMin = (deltaTime // 60) % 60
  deltaTimeHour = (deltaTime // 3600) % 24

  # Calculate the elapsed time per item (population size * number of iterations).
  deltaTimePerItem = deltaTime / float(popSize * numIter)
  deltaTimePerItemSec = deltaTimePerItem % 60
  deltaTimePerItemMin = (deltaTimePerItem // 60) % 60
  deltaTimePerItemHour = (deltaTimePerItem // 3600) % 24

  if (doVerbose):
    print(f"Elapsed Time: {deltaTimeHour}h {deltaTimeMin}m {deltaTimeSec:.2f}s")
    print(f"Elapsed Time (Per Item): {deltaTimePerItemHour}h {deltaTimePerItemMin}m {deltaTimePerItemSec:.2f}s")

  if (doStore):
    # Store the elapsed time in a text file.
    with open(os.path.join(storageDir, f"{keyword}-ElapsedTime.txt"), "w") as f:
      f.write(f"Elapsed Time: {deltaTime}\n")
      f.write(f"Elapsed Time (h): {deltaTimeHour}\n")
      f.write(f"Elapsed Time (m): {deltaTimeMin}\n")
      f.write(f"Elapsed Time (s): {deltaTimeSec}\n")
      f.write(f"Elapsed Time (Per Item): {deltaTimePerItem}\n")
      f.write(f"Elapsed Time (Per Item) (h): {deltaTimePerItemHour}\n")
      f.write(f"Elapsed Time (Per Item) (m): {deltaTimePerItemMin}\n")
      f.write(f"Elapsed Time (Per Item) (s): {deltaTimePerItemSec}")

  return {
    "deltaTime"           : deltaTime,
    "deltaTimeSec"        : deltaTimeSec,
    "deltaTimeMin"        : deltaTimeMin,
    "deltaTimeHour"       : deltaTimeHour,
    "deltaTimePerItem"    : deltaTimePerItem,
    "deltaTimePerItemSec" : deltaTimePerItemSec,
    "deltaTimePerItemMin" : deltaTimePerItemMin,
    "deltaTimePerItemHour": deltaTimePerItemHour,
  }


def PlotStoreConvergenceCurve(histBestFit, modelStr, argsDict, keyword, extensions=["png", "pdf"]):
  """
  Plot and store the convergence curve of the optimization process.

  Parameters:
  ----------
  histBestFit : list or numpy.ndarray
      List of best fitness values over iterations.
  modelStr : str
      Name of the machine learning model used for evaluation.
  argsDict : dict
      Dictionary containing relevant parameters such as storage directory and keyword.
  extensions : list, optional
      File formats in which to save the plot (default: ["png", "pdf"]).

  This function generates a convergence curve plot and saves it in the specified directory.
  """
  storageDir = argsDict["storageDir"]

  # Plot the convergence curve for the optimization process.
  plt.figure()
  plt.plot(histBestFit, lw=2)
  plt.title(f"Convergence Curve for {modelStr}")
  plt.xlabel("Iteration")
  plt.ylabel("Fitness")
  plt.grid()
  plt.tight_layout()

  # Save the plot in the specified file formats.
  for ext in extensions:
    plt.savefig(
      os.path.join(storageDir, f"{keyword}-ConvergenceCurve.{ext}"),
      dpi=500,
      bbox_inches="tight",
    )

  plt.close()  # Close the plot to free memory.
  plt.clf()


def StoreVerboseDetails(historyPath, argsDict, keyword):
  """
  Store detailed results of the optimization process, including top features and intermediate calculations.

  Parameters:
  ----------
  argsDict : dict
      Dictionary containing relevant parameters such as storage directory, keyword, and top features count.

  This function processes the optimization history to extract and store top features, averages, and other details.
  """
  storageDir = argsDict["storageDir"]
  top = argsDict["top"]
  storeDetailedFiles = argsDict["storeDetailedFiles"]

  # Retrieve the best results from the optimization process.
  df = pd.read_csv(historyPath)

  # Get the columns starting with "X" and having more than one character.
  columns = [el for el in df.columns if el.startswith("X") and len(el) > 1] + ["Fitness"]

  # Extract the best results based on the selected columns.
  portion = df[columns]

  # Check whether to store detailed files.
  if (storeDetailedFiles):
    # Store the best results in a new DataFrame.
    portion.to_csv(os.path.join(storageDir, f"{keyword}-Portion.csv"), index=False)

  # Group the portion by the "X" columns and calculate the average fitness for each group.
  groupByX = portion.groupby("X (Str)")
  avg = groupByX.mean().reset_index(drop=True)
  avg = avg.sort_values(by="Fitness", ascending=False).reset_index(drop=True)

  # Check whether to store detailed files.
  if (storeDetailedFiles):
    # Store the average fitness values in a new DataFrame.
    avg.to_csv(os.path.join(storageDir, f"{keyword}-Averages.csv"), index=False)

  # Get the top features based on the average fitness values.
  topFeatures = avg.head(top)

  # Check whether to store detailed files.
  if (storeDetailedFiles):
    # Store the top features in a new DataFrame.
    topFeatures.to_csv(os.path.join(storageDir, f"{keyword}-TopFeatures.csv"), index=False)

  # Perform intermediate calculations to derive final feature rankings.
  step1 = topFeatures.drop(columns=["Fitness"]).sum(axis=1).values
  count = topFeatures.drop(columns=["Fitness"]).count(axis=1).values
  cntFitness = topFeatures["Fitness"].values
  step2 = 1.0 - 0.5 * (step1 / count)
  newTopFeatures = topFeatures.copy()
  newTopFeatures.drop(columns=["Fitness"], inplace=True)
  newTopFeatures = newTopFeatures * step2[:, None] * cntFitness[:, None]
  step4 = newTopFeatures.astype(bool).sum(axis=0).values
  step5 = newTopFeatures.sum(axis=0).values
  step6 = step4 / float(top)
  step7 = step5 * step6

  # Sort the final fitness values in descending order and get the sorted indices.
  indices = np.argsort(step7)[::-1]
  sortedTopFeatures = topFeatures.iloc[:, indices]
  sortedTopFeatures["Fitness"] = topFeatures["Fitness"]

  # Check whether to store detailed files.
  if (storeDetailedFiles):
    # Store the final top features in a new DataFrame.
    sortedTopFeatures.to_csv(os.path.join(storageDir, f"{keyword}-SortedTopFeatures.csv"), index=False)

  # Print the column names of the top features.
  L = topFeatures.iloc[:, indices].columns.tolist()
  print("Top Features:", L)

  # Create a detailed DataFrame for intermediate steps.
  details = newTopFeatures.copy()
  details.insert(0, "Step 1", step1)
  details.insert(1, "Step 2", step2)
  details.loc["Step 4"] = ["", "Step 4"] + list(step4)
  details.loc["Step 5"] = ["", "Step 5"] + list(step5)
  details.loc["Step 6"] = ["", "Step 6"] + list(step6)
  details.loc["Step 7"] = ["", "Step 7"] + list(step7)
  details.loc["Sorted"] = ["", "Sorted"] + step7[indices].tolist()
  details.loc["Ranking"] = ["", "Ranking"] + L

  # Check whether to store detailed files.
  if (storeDetailedFiles):
    # Store the details in a new CSV file.
    details.to_csv(os.path.join(storageDir, f"{keyword}-StepsDetails.csv"), index=False)

  # Store the top features in a text file.
  with open(os.path.join(storageDir, f"{keyword}-TopFeaturesNames.txt"), "w") as f:
    f.write("Horizontal Display (List):" + "\n")
    f.write("[" + ", ".join([str(x) for x in L]) + "]\n")
    f.write("Vertical Display:" + "\n")
    f.write("\n".join(L))

  # Store the top features in a text file with adjusted indexing.
  with open(os.path.join(storageDir, f"{keyword}-TopFeaturesNumbers.txt"), "w") as f:
    f.write("Displaying Indices:" + "\n")
    f.write("Features Indices Starting from Index 1:\n[" + ", ".join([str(x[1:]) for x in L]) + "]\n")
    f.write("Features Indices Starting from Index 0:\n[" + ", ".join([str((int(x[1:]) - 1)) for x in L]) + "]\n")

    print("Displaying Indices:")
    print("Features Indices Starting from Index 1: [" + ", ".join([str(x[1:]) for x in L]) + "]")
    print("Features Indices Starting from Index 0: [" + ", ".join([str((int(x[1:]) - 1)) for x in L]) + "]")
