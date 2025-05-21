# Import necessary libraries.
import IgnoreWarnings
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
  RandomForestClassifier,
  AdaBoostClassifier,
  GradientBoostingClassifier,
  ExtraTreesClassifier,
  BaggingClassifier,
)
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import (
  normalize,
  StandardScaler,
  MinMaxScaler,
  MaxAbsScaler,
  RobustScaler,
  LabelEncoder,
)
from imblearn.over_sampling import (
  SMOTE,
  ADASYN,
  RandomOverSampler,
  BorderlineSMOTE,
  KMeansSMOTE,
  SVMSMOTE,
)
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from PerformanceMetrics import CalculateAllMetricsUpdated


class FeatureSelectionMeta(object):
  """
  A class to perform feature selection using metaheuristic algorithms.

  This class integrates various metaheuristic optimization algorithms (e.g., MRFO, PSO, GWO) with machine learning
  models to identify the optimal subset of features for a given dataset. It supports preprocessing steps such as
  normalization, sampling, and encoding, and evaluates solutions based on performance metrics derived from a confusion matrix.

  Parameters:
  ----------
  datasetPath : str
      Path to the dataset file.
  meta : str, optional
      Metaheuristic algorithm to use (default: "MRFO").
  historyPath : str, optional
      Path to the history file for logging optimization results.
  ps : int, optional
      Population size for the metaheuristic algorithm (default: 10).
  T : int, optional
      Number of iterations for the metaheuristic algorithm (default: 100).
  isMax : bool, optional
      Whether to maximize or minimize the fitness function (default: True).
  fillValue : float, optional
      Value to fill missing data in the dataset (default: 0.0).
  testSize : float, optional
      Proportion of the dataset to include in the test split (default: 0.2).
  doSplit : bool, optional
      Whether to split the dataset into training and testing sets (default: True).
  forceSplit : bool, optional
      Whether to force the dataset to be split into training and testing sets (default: True).
  normTechnique : str, optional
      Normalization technique to apply to the dataset.
  sampleTechnique : str, optional
      Sampling technique to handle class imbalance.
  modelStr : str, optional
      Name of the machine learning model to use for evaluation (default: "RandomForestClassifier").
  hyperparams : dict, optional
      Hyperparameters for the machine learning model.
  loggingPeriod : int, optional
      Frequency of logging the optimization process (default: 1).
  prefix : str, optional
      Prefix for log files.
  storageDir : str, optional
      Directory to store the optimization results (default: "Results").

  Methods:
  -------
  Optimize()
      Run the metaheuristic algorithm to optimize feature selection.
  PreprocessData()
      Preprocess the dataset by loading, splitting, encoding, normalizing, and sampling.
  EvaluateFitness(x)
      Evaluate the fitness of a given solution.
  LoadDataset()
      Load and preprocess the dataset.
  EncodeData()
      Encode categorical data using LabelEncoder.
  SplitData()
      Split the dataset into training and testing sets.
  NormalizeData()
      Normalize the dataset using the specified technique.
  SampleDataset()
      Apply the specified sampling technique to handle class imbalance.
  """

  def __init__(
    self,
    datasetPath,
    meta="MRFO",
    historyPath=None,
    ps=10,
    T=100,
    isMax=True,
    fillValue=0.0,
    testSize=0.2,
    doSplit=True,
    forceSplit=True,
    normTechnique=None,
    sampleTechnique=None,
    modelStr="RandomForestClassifier",
    hyperparams={},
    loggingPeriod=1,
    prefix=None,
    storageDir="Results",
  ):
    """
    Initialize the FeatureSelectionMeta class.

    Parameters:
    ----------
    datasetPath : str
        Path to the dataset file.
    meta : str, optional
        Metaheuristic algorithm to use (default: "MRFO").
    historyPath : str, optional
        Path to the history file for logging optimization results.
    ps : int, optional
        Population size for the metaheuristic algorithm (default: 10).
    T : int, optional
        Number of iterations for the metaheuristic algorithm (default: 100).
    isMax : bool, optional
        Whether to maximize or minimize the fitness function (default: True).
    fillValue : float, optional
        Value to fill missing data in the dataset (default: 0.0).
    testSize : float, optional
        Proportion of the dataset to include in the test split (default: 0.2).
    doSplit : bool, optional
        Whether to split the dataset into training and testing sets (default: True).
    forceSplit : bool, optional
        Whether to force the dataset to be split into training and testing sets (default: True).
    normTechnique : str, optional
        Normalization technique to apply to the dataset.
    sampleTechnique : str, optional
        Sampling technique to handle class imbalance.
    modelStr : str, optional
        Name of the machine learning model to use for evaluation (default: "RandomForestClassifier").
    hyperparams : dict, optional
        Hyperparameters for the machine learning model.
    loggingPeriod : int, optional
        Frequency of logging the optimization process (default: 1).
    prefix : str, optional
        Prefix for log files.
    storageDir : str, optional
        Directory to store the optimization results (default: "Results").
    """
    # Initialize the prefix for log files.
    self.prefix = prefix

    # Initialize paths and parameters.
    self.datasetPath = datasetPath
    self.historyPath = historyPath
    self.meta = meta
    self.basePath = os.path.basename(self.datasetPath)
    self.trainPath = os.path.join(storageDir, self.basePath.replace(".csv", f"_Train_Subset.csv"))
    self.testPath = os.path.join(storageDir, self.basePath.replace(".csv", f"_Test_Subset.csv"))

    # Optimization problem parameters.
    self.D = None
    self.lb = 0.0
    self.ub = 1.0
    self.ps = ps
    self.T = T
    self.isMax = isMax
    self.fillValue = fillValue

    # Dataset splitting parameters.
    self.testSize = testSize
    self.doSplit = doSplit
    self.forceSplit = forceSplit

    # Preprocessing techniques.
    self.normTechnique = normTechnique
    self.sampleTechnique = sampleTechnique

    # Machine learning model parameters.
    self.modelStr = modelStr
    self.hyperparams = hyperparams

    # Logging parameters.
    self.loggingPeriod = loggingPeriod
    self.storageDir = storageDir
    self.history = []
    self.periodCounter = 0

    # Load history if it exists.
    if (os.path.exists(self.historyPath)):
      df = pd.read_csv(self.historyPath)
      self.history = df.to_dict("records")

    # Print initialization details.
    print("Meta Information:")
    print(f"- Dataset Path: {self.datasetPath}")
    print(f"- Metaheuristic: {self.meta}")
    print(f"- Train Path: {self.trainPath}")
    print(f"- Test Path: {self.testPath}")
    print(f"- Population Size: {self.ps}")
    print(f"- Number of Iterations: {self.T}")
    print(f"- Maximization: {self.isMax}")
    print(f"- Fill Value: {self.fillValue}")
    print(f"- Test Size: {self.testSize}")
    print(f"- Split Dataset: {self.doSplit}")
    print(f"- Normalization Technique: {self.normTechnique}")
    print(f"- Sampling Technique: {self.sampleTechnique}")
    print(f"- Model Name: {self.modelStr}")
    print(f"- All Hyperparameters: {self.hyperparams}")
    print(f"- Logging Period: {self.loggingPeriod}")
    print(f"- Log Path: {self.historyPath}")
    print(f"- History Length: {len(self.history)}")

  def Optimize(self):
    """
    Run the metaheuristic algorithm to optimize feature selection.

    This method initializes the selected metaheuristic algorithm, runs the optimization process,
    and logs the results to a CSV file.
    """
    # Preprocess the dataset before optimization.
    self.PreprocessData()

    # Initialize the selected metaheuristic algorithm.
    if (self.meta == "MRFO"):
      from MantaRayForagingAlgorithm import MantaRayForagingAlgorithm

      self.metaObj = MantaRayForagingAlgorithm(
        self.EvaluateFitness, self.lb, self.ub, self.D, self.ps, self.T, self.isMax
      )
    elif (self.meta == "PSO"):
      from ParticleSwarmOptimization import ParticleSwarmOptimization

      self.metaObj = ParticleSwarmOptimization(
        self.EvaluateFitness, self.lb, self.ub, self.D, self.ps, self.T, self.isMax
      )
    elif (self.meta == "GWO"):
      from GreyWolfOptimizer import GreyWolfOptimizer

      self.metaObj = GreyWolfOptimizer(
        self.EvaluateFitness, self.lb, self.ub, self.D, self.ps, self.T, self.isMax
      )
    else:
      raise Exception("Metaheuristic not found.")

    # Run the optimization process.
    self.metaObj.Optimize()

    # Save the optimization history to a CSV file.
    df = pd.DataFrame(self.history)
    df.sort_values("Fitness", ascending=not self.isMax, inplace=True)
    df.to_csv(self.historyPath, index=False)

    # Print the best fitness value and solution found during optimization.
    print(f"Best Fitness: {self.metaObj.bestF}")
    print(f"Best Solution: {list(self.metaObj.bestX)}")

  def PreprocessData(self):
    """
    Preprocess the dataset by loading, splitting, encoding, normalizing, and sampling.

    This method prepares the dataset for feature selection by applying various preprocessing steps.
    """
    # Load the dataset.
    self.LoadDataset()

    # Split the dataset into training and testing sets.
    self.SplitData()

    # Encode categorical data using LabelEncoder.
    self.EncodeData()

    # Normalize the dataset using the specified technique.
    self.NormalizeData()

    # Apply the specified sampling technique to handle class imbalance.
    self.SampleDataset()

    # Set the dimension of the problem (number of features).
    self.D = self.xTrain.shape[1]

    # Print information about the preprocessed data.
    print("Data Preprocessing:")
    print(f"- Dataset Shape: {self.X.shape}")
    print(f"- Features Shape: {self.featuresColumns}")
    print(f"- Target Shape: {self.targetColumn}")
    print(f"- Encoded Target: {self.encoder.classes_}")
    print(f"- Train Shape: {self.xTrain.shape}")
    print(f"- Test Shape: {self.xTest.shape}")
    print(f"- Dimension of the Problem: {self.D}")

  def EvaluateFitness(self, x):
    """
    Evaluate the fitness of a given solution.

    This method converts the solution vector to binary, selects the corresponding features,
    trains a machine learning model, and evaluates its performance using a confusion matrix.

    Parameters:
    ----------
    x : numpy.ndarray
        The solution vector representing feature selection.

    Returns:
    -------
    float
        The fitness score (SCORE) for the current solution.
    """
    # Define a dictionary of available machine learning models.
    self.models = {
      "RF"     : RandomForestClassifier(),
      "DT"     : DecisionTreeClassifier(),
      "SVC"    : SVC(),
      "KNN"    : KNeighborsClassifier(),
      "MLP"    : MLPClassifier(),
      "AB"     : AdaBoostClassifier(),
      "GB"     : GradientBoostingClassifier(),
      "ETs"    : ExtraTreesClassifier(),
      "Bagging": BaggingClassifier(),
      "XGB"    : XGBClassifier(),
    }

    # Convert continuous values in the solution vector to binary (0 or 1).
    int2bool = lambda x: 1 if (x >= 0.5) else 0
    x = [int2bool(xi) for xi in x]
    x = np.array(x)

    # Find the indices of selected features (where x == 1).
    idx = np.where(x == 1)[0]
    if (len(idx) == 0):
      return 0.0

    xTrain = self.xTrain
    xTest = self.xTest

    # Check if no normalization nor sampling is applied.
    if (self.normTechnique is None and self.sampleTechnique is None):
      # Convert the training and testing datasets to numpy arrays.
      xTrain = xTrain.values
      xTest = xTest.values

    # Select the corresponding features from the training and testing datasets.
    xTrain = xTrain[:, idx]
    xTest = xTest[:, idx]

    # Initialize the selected machine learning model.
    model = self.models[self.modelStr]
    model = model.__class__()

    # Randomly pick hyperparameters for the model.
    pickedHyperparams = {}
    for key, value in self.hyperparams.items():
      pickedHyperparams[key] = value[np.random.randint(0, len(value))]

    # Set the selected hyperparameters for the model.
    model.set_params(**pickedHyperparams)

    # Train the model on the selected features.
    model.fit(xTrain, self.yTrain)

    # Predict the target variable for the test set.
    predTest = model.predict(xTest)

    # Calculate the confusion matrix for the predictions.
    cm = confusion_matrix(self.yTest, predTest)

    # Calculate all performance metrics based on the confusion matrix.
    metrics = CalculateAllMetricsUpdated(cm)

    # Append the results to the optimization history.
    self.history.append(
      {
        "Model"        : self.modelStr,
        "Meta"         : self.meta,
        "Normalization": self.normTechnique,
        "Sampling"     : self.sampleTechnique,
        **{f"X{i + 1}": xi for i, xi in enumerate(x)},
        "Fitness"      : metrics["SCORE"],
        **metrics,
        "X (Str)"      : "'" + "".join(map(str, x)),
        "X"            : x,
        "Hyper"        : pickedHyperparams,
      }
    )

    # Increment the logging period counter.
    self.periodCounter += 1

    # Save the optimization history to a CSV file if the logging period is reached.
    if (self.periodCounter >= self.loggingPeriod):
      self.periodCounter = 0
      df = pd.DataFrame(self.history)
      df.sort_values("Fitness", ascending=not self.isMax, inplace=True)
      df.to_csv(self.historyPath, index=False)

    # Return the fitness score (SCORE) for the current solution.
    return float(metrics["SCORE"])

  def LoadDataset(self):
    """
    Load and preprocess the dataset.

    This method loads the dataset, shuffles it, fills missing values, and splits it into features and target.
    """
    # Load the dataset from the specified file path.
    data = pd.read_csv(self.datasetPath)

    # Shuffle the dataset to ensure randomness.
    data = data.sample(frac=1).reset_index(drop=True)

    # Fill missing values in the dataset with the specified fill value.
    data = data.fillna(self.fillValue)

    # Extract the feature columns (all columns except the last one).
    featuresColumns = data.columns[:-1]

    # Extract the target column (the last column).
    targetColumn = data.columns[-1]

    # Split the dataset into features (X) and target (y).
    X = data[featuresColumns]
    y = data[targetColumn]

    # Get the column names of the features that are not numerical.
    self.nonNumericalColumns = X.select_dtypes(exclude=np.number).columns

    # Store the features and target in the class attributes.
    self.X = X
    self.y = y
    self.featuresColumns = featuresColumns
    self.targetColumn = targetColumn

  def EncodeData(self):
    """
    Encode categorical data using LabelEncoder.

    This method encodes the target variable and non-numerical features using LabelEncoder.
    """
    # Initialize a LabelEncoder for encoding the target variable.
    encoder = LabelEncoder()

    # Encode the target variable (y) using the LabelEncoder.
    self.y = encoder.fit_transform(self.y)

    # Encode the training target variable (yTrain).
    self.yTrain = encoder.transform(self.yTrain)

    # Encode the testing target variable (yTest).
    self.yTest = encoder.transform(self.yTest)

    # Store the LabelEncoder object in the class attributes.
    self.encoder = encoder

    # Encode the non-numerical columns in the features.
    for col in self.nonNumericalColumns:
      # Initialize a LabelEncoder for encoding the non-numerical column.
      encoder = LabelEncoder()

      # Encode the non-numerical column using the LabelEncoder.
      self.X[col] = encoder.fit_transform(self.X[col])

      # Encode the training non-numerical column.
      self.xTrain[col] = encoder.transform(self.xTrain[col])

      # Encode the testing non-numerical column.
      self.xTest[col] = encoder.transform(self.xTest[col])

      # Store the LabelEncoder object in the class attributes.
      self.encoder = encoder

  def SplitData(self):
    """
    Split the dataset into training and testing sets.

    This method checks if the dataset should be split and ensures that the split is reproducible.
    """
    # Check if the dataset should be split into training and testing sets.
    if (self.doSplit or self.forceSplit):
      # Check if the training and testing datasets already exist.
      exist = os.path.exists(self.trainPath) and os.path.exists(self.testPath)
      if (not exist or self.forceSplit):
        # Split the dataset into training and testing sets.
        xTrain, xTest, yTrain, yTest = train_test_split(
          self.X,
          self.y,
          test_size=self.testSize,  # Proportion of the dataset to include in the test split.
          random_state=np.random.randint(0, 100),  # Random seed for reproducibility.
          stratify=self.y,  # Ensure that the class distribution is preserved.
        )

        # Store the training and testing datasets in the class attributes.
        self.xTrain = xTrain
        self.xTest = xTest
        self.yTrain = yTrain
        self.yTest = yTest

        # Combine the features and target for the training set.
        self.trainDf = pd.concat([self.xTrain, self.yTrain], axis=1)

        # Combine the features and target for the testing set.
        self.testDf = pd.concat([self.xTest, self.yTest], axis=1)

        # Save the training dataset to a CSV file.
        self.trainDf.to_csv(self.trainPath, index=False)

        # Save the testing dataset to a CSV file.
        self.testDf.to_csv(self.testPath, index=False)

    # Load the training dataset from the specified file path.
    self.trainDF = pd.read_csv(self.trainPath)

    # Load the testing dataset from the specified file path.
    self.testDF = pd.read_csv(self.testPath)

    # Extract the feature columns from the training dataset.
    features = self.trainDF.columns[:-1]

    # Extract the target column from the training dataset.
    target = self.trainDF.columns[-1]

    # Store the training features and target in the class attributes.
    self.xTrain = self.trainDF[features]
    self.yTrain = self.trainDF[target]

    # Store the testing features and target in the class attributes.
    self.xTest = self.testDF[features]
    self.yTest = self.testDF[target]

  def NormalizeData(self):
    """
    Normalize the dataset using the specified technique.

    This method applies normalization techniques such as L1, L2, MinMax scaling, etc.
    """
    # Define a dictionary of normalization techniques.
    techniques = {
      "Normalize_L1_0" : lambda x: normalize(x, norm="l1", axis=0),
      "Normalize_L2_0" : lambda x: normalize(x, norm="l2", axis=0),
      "Normalize_Max_0": lambda x: normalize(x, norm="max", axis=0),
      "Normalize_L1_1" : lambda x: normalize(x, norm="l1", axis=1),
      "Normalize_L2_1" : lambda x: normalize(x, norm="l2", axis=1),
      "Normalize_Max_1": lambda x: normalize(x, norm="max", axis=1),
      "StandardScaler" : StandardScaler(),
      "MinMaxScaler"   : MinMaxScaler(),
      "MaxAbsScaler"   : MaxAbsScaler(),
      "RobustScaler"   : RobustScaler(),
    }

    # Check if a normalization technique is specified.
    if (self.normTechnique is None):
      return

    # Apply normalization if the technique starts with "Normalize".
    if (self.normTechnique.startswith("Normalize")):
      self.xTrain = techniques[self.normTechnique](self.xTrain)
      self.xTest = techniques[self.normTechnique](self.xTest)
    else:
      # Initialize the normalization object.
      normObj = techniques[self.normTechnique]

      # Fit and transform the training data using the normalization technique.
      self.xTrain = normObj.fit_transform(self.xTrain)

      # Transform the testing data using the normalization technique.
      self.xTest = normObj.transform(self.xTest)

  def SampleDataset(self):
    """
    Apply the specified sampling technique to handle class imbalance.

    This method uses oversampling, undersampling, or combined techniques to balance the dataset.
    """
    # Define a dictionary of sampling techniques.
    techniques = {
      "SMOTE"             : SMOTE,
      "ADASYN"            : ADASYN,
      "RandomOverSampler" : RandomOverSampler,
      "RandomUnderSampler": RandomUnderSampler,
      "NearMiss"          : NearMiss,
      "TomekLinks"        : TomekLinks,
      "SMOTETomek"        : SMOTETomek,
      "SMOTEENN"          : SMOTEENN,
      "SVMSMOTE"          : SVMSMOTE,
      "KMeansSMOTE"       : KMeansSMOTE,
      "BorderlineSMOTE"   : BorderlineSMOTE,
    }

    # Check if a sampling technique is specified.
    if (self.sampleTechnique is None):
      return

    # Initialize the sampling object.
    sampleObj = techniques[self.sampleTechnique]()

    # Fit and resample the training data using the sampling technique.
    self.xTrain, self.yTrain = sampleObj.fit_resample(self.xTrain, self.yTrain)
