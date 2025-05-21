# Import necessary libraries.
import IgnoreWarnings
import argparse
import os
import time

from FeatureSelectionMeta import FeatureSelectionMeta
from Configs import Configs
from Helpers import *

# Create an instance of the Configs class to access predefined hyperparameters.
configs = Configs()

# Use argparse to parse command-line arguments.
parser = argparse.ArgumentParser(description="H2LFR by Hossam Magdy Balaha")
parser.add_argument(
  "--dataset", type=str, required=True,
  help="Path to the dataset file."
)
parser.add_argument(
  "--storage", type=str, required=False,
  help="Directory to store the results."
)
parser.add_argument(
  "--doSplit", type=bool, required=False, default=True,
  help="Whether to split the dataset into training and testing sets."
)
parser.add_argument(
  "--forceSplit", type=bool, required=False, default=True,
  help="Whether to force split the dataset into training and testing sets."
)
parser.add_argument(
  "--popSize", type=int, required=False, default=50,
  help="Population size for the metaheuristic algorithm."
)
parser.add_argument(
  "--numIter", type=int, required=False, default=50,
  help="Number of iterations for the metaheuristic algorithm."
)
parser.add_argument(
  "--isMax", type=bool, required=False, default=True,
  help="Whether to maximize or minimize the fitness function."
)
parser.add_argument(
  "--testSize", type=float, required=False, default=0.2,
  help="Test size for the train-test split."
)
parser.add_argument(
  "--fillValue", type=float, required=False, default=0.0,
  help="Fill value for missing data."
)
parser.add_argument(
  "--normTechnique", type=str, required=False, default=None,
  help="Normalization technique to be applied."
)
parser.add_argument(
  "--sampleTechnique", type=str, required=False, default=None,
  help="Sampling technique to be applied."
)
parser.add_argument(
  "--loggingPeriod", type=int, required=False, default=50,
  help="Logging period for saving optimization history."
)
parser.add_argument(
  "--top", type=int, required=False, default=15,
  help="Number of top features to select."
)
parser.add_argument(
  "--meta", type=str, required=False, default="MRFO",
  help="Meta-heuristic algorithm to be used."
)
parser.add_argument(
  "--modelsStrList", type=str, required=False, default="'DT','KNN'",
  help="Models string to be used (comma-separated)."
)
parser.add_argument(
  "--storeDetailedFiles", type=bool, required=False, default=False,
  help="Whether to store the detailed files."
)

# Get the current timestamp.
timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

# Parse the command-line arguments.
args = parser.parse_args()

# Assign parsed arguments to variables for easier access.
datasetPath = args.dataset.strip()
if (args.storage):
  storageDir = args.storage.strip()
else:
  datasetContainer = os.path.dirname(datasetPath)
  folderOnly = os.path.basename(datasetContainer)
  storageDir = folderOnly + " Results"

doSplit = args.doSplit
forceSplit = args.forceSplit
popSize = args.popSize
numIter = args.numIter
isMax = args.isMax
testSize = args.testSize
fillValue = args.fillValue
normTechnique = args.normTechnique
sampleTechnique = args.sampleTechnique
loggingPeriod = args.loggingPeriod
top = args.top
meta = args.meta
storeDetailedFiles = args.storeDetailedFiles
modelsStrList = args.modelsStrList
modelsStrList = [
  item.replace("'", "").replace('"', "")
  for item in modelsStrList.split(",")
]

# Define the directory to store the optimization results.
storageDir = os.path.join(timestamp, storageDir)
# Create the storage directory if it does not exist.
os.makedirs(storageDir, exist_ok=True)

# Extract the base name of the dataset file.
fileName = os.path.basename(datasetPath)
basePathNoExt = os.path.splitext(fileName)[0]

# Collect the arguments in a dictionary to store in a file.
argsDict = {
  "datasetPath"       : datasetPath,
  "storageDir"        : storageDir,
  "doSplit"           : doSplit,
  "forceSplit"        : forceSplit,
  "popSize"           : popSize,
  "numIter"           : numIter,
  "isMax"             : isMax,
  "testSize"          : testSize,
  "fillValue"         : fillValue,
  "normTechnique"     : normTechnique,
  "sampleTechnique"   : sampleTechnique,
  "loggingPeriod"     : loggingPeriod,
  "top"               : top,
  "meta"              : meta,
  "modelsStrList"     : modelsStrList,
  "storeDetailedFiles": storeDetailedFiles,
  "basePathNoExt"     : basePathNoExt,
  "timestamp"         : timestamp,
}

# Store the arguments in a text file for reference.
StoreArgumentsToText(storageDir, basePathNoExt, argsDict)

# Iterate through the list of models specified in the command-line arguments.
for i, modelStr in enumerate(modelsStrList):
  print("=" * 50)
  print(f"Model: {modelStr}")

  # Generate a keyword for the current model based on its parameters.
  keyword = f"{meta}-{modelStr}"
  if (normTechnique is not None):
    keyword += f"-{normTechnique}"
  if (sampleTechnique is not None):
    keyword += f"-{sampleTechnique}"

  # Define the path for the history file.
  historyPath = os.path.join(storageDir, f"{keyword}-History.csv")

  # Retrieve the hyperparameters for the current model from the `Configs` class.
  hyperparams = configs.modelsHyperparameters[modelStr]

  # Initialize the FeatureSelectionMeta object with the specified parameters.
  obj = FeatureSelectionMeta(
    datasetPath=datasetPath, meta=meta, historyPath=historyPath,
    ps=popSize, T=numIter, isMax=isMax, fillValue=fillValue,
    testSize=testSize, doSplit=doSplit, forceSplit=forceSplit,
    normTechnique=normTechnique, sampleTechnique=sampleTechnique,
    modelStr=modelStr, hyperparams=hyperparams,
    loggingPeriod=loggingPeriod, prefix=basePathNoExt,
    storageDir=storageDir,
  )

  # Run the optimization process using the metaheuristic algorithm.
  startTime = time.time()
  obj.Optimize()
  endTime = time.time()

  # Calculate and store the elapsed time for the optimization process.
  CalculateElapsedTime(startTime, endTime, argsDict, keyword, doStore=True, doVerbose=True)

  # Retrieve the best fitness values from the optimization history.
  histBestFit = obj.metaObj.histBestFit

  # Retrieve the population and fitness values from the optimization process.
  population = obj.metaObj.pop
  fitness = obj.metaObj.fitness

  # Convert continuous values in the population to binary (0 or 1).
  int2bool = lambda x: 1 if (x >= 0.5) else 0
  population = [[int2bool(x) for x in individual] for individual in population]

  # Plot and store the convergence curve for the optimization process.
  PlotStoreConvergenceCurve(histBestFit, modelStr, argsDict, keyword, extensions=["png", "pdf"])

  # Store verbose details of the optimization process.
  StoreVerboseDetails(historyPath, argsDict, keyword)

# Print a message indicating that the process has completed.
print("Completed!")
