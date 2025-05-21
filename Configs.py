class Configs(object):
  """
  A class to define hyperparameter configurations for various machine learning models.

  This class provides a centralized configuration for hyperparameters used in different machine learning models.
  Each model's hyperparameters are stored as a dictionary, where the keys represent the hyperparameter names,
  and the values are lists of possible values for tuning or optimization.

  Attributes:
  ----------
  modelsHyperparameters : dict
      A dictionary containing hyperparameter configurations for various machine learning models.
      Each key corresponds to a model name (e.g., "RF" for Random Forest), and the value is another
      dictionary specifying the hyperparameters and their possible values.

  Supported Models:
  -----------------
  - RF: Random Forest Classifier.
  - DT: Decision Tree Classifier.
  - SVC: Support Vector Classifier.
  - KNN: K-Nearest Neighbors Classifier.
  - MLP: Multi-Layer Perceptron Classifier.
  - AB: AdaBoost Classifier.
  - GB: Gradient Boosting Classifier.
  - ETs: Extra Trees Classifier.
  - Bagging: Bagging Classifier.
  - XGB: XGBoost Classifier.

  Example Usage:
  --------------
  configs = Configs()
  rfHyperparams = configs.modelsHyperparameters["RF"]
  """

  def __init__(self):
    """
    Initialize the Configs class with predefined hyperparameter configurations for various models.
    """
    self.modelsHyperparameters = {
      "RF"     : {
        "n_estimators"     : [10, 50, 100, 200, 500],
        "max_features"     : ["sqrt", "log2", None],
        "max_depth"        : [None, 10, 50, 100, 200, 500, 1000],
        "min_samples_split": [2, 5, 10, 50, 100, 200],
        "min_samples_leaf" : [1, 2, 5, 10, 50, 100],
        "bootstrap"        : [True, False],
      },
      "DT"     : {
        "max_depth"        : [None, 10, 50, 100, 200, 500, 1000],
        "min_samples_split": [2, 5, 10, 50, 100, 200],
        "min_samples_leaf" : [1, 2, 5, 10, 50, 100],
        "max_features"     : ["sqrt", "log2", None],
        "splitter"         : ["best", "random"],
      },
      "SVC"    : {
        "C"                      : [0.01, 0.1, 1, 10, 100],
        "kernel"                 : ["linear", "poly", "rbf", "sigmoid"],
        "degree"                 : [2, 3, 4, 5, 6],
        "gamma"                  : ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        "coef0"                  : [0.0, 0.1, 0.5, 1.0, 2.0],
        "shrinking"              : [True, False],
        "probability"            : [True, False],
        "class_weight"           : [None, "balanced"],
        "decision_function_shape": ["ovo", "ovr"],
      },
      "KNN"    : {
        "n_neighbors": [3, 5, 7, 10, 15, 20],
        "weights"    : ["uniform", "distance"],
        "algorithm"  : ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size"  : [10, 20, 30, 40, 50, 60],
        "p"          : [1, 2],
        "metric"     : ["minkowski", "euclidean", "manhattan"],
      },
      "MLP"    : {
        "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100), (100, 100, 50), (100, 100, 100)],
        "activation"        : ["identity", "logistic", "tanh", "relu"],
        "solver"            : ["lbfgs", "sgd", "adam"],
        "alpha"             : [0.0001, 0.001, 0.01, 0.1],
        "learning_rate"     : ["constant", "invscaling", "adaptive"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter"          : [200, 300, 400, 500, 600],
        "early_stopping"    : [True, False],
      },
      "AB"     : {
        "n_estimators" : [10, 50, 100, 200, 500, 1000],
        "learning_rate": [0.01, 0.1, 1, 10],
        "algorithm"    : ["SAMME", "SAMME.R"],
      },
      "GB"     : {
        "n_estimators"     : [10, 50, 100, 200, 500, 1000],
        "learning_rate"    : [0.01, 0.1, 0.5, 1, 10],
        "max_depth"        : [3, 5, 7, 9],
        "min_samples_split": [2, 5, 10, 50, 100],
        "min_samples_leaf" : [1, 2, 5, 10, 50, 100],
        "subsample"        : [0.5, 0.7, 1.0],
        "max_features"     : ["auto", "sqrt", "log2", None],
      },
      "ETs"    : {
        "n_estimators"     : [10, 50, 100, 200, 500],
        "max_features"     : ["sqrt", "log2", None],
        "max_depth"        : [None, 10, 50, 100, 200, 500, 1000],
        "min_samples_split": [2, 5, 10, 50, 100, 200],
        "min_samples_leaf" : [1, 2, 5, 10, 50, 100],
        "bootstrap"        : [True, False],
      },
      "Bagging": {
        "n_estimators"      : [10, 50, 100, 200, 500, 1000],
        "max_samples"       : [0.1, 0.5, 1.0],
        "max_features"      : [0.1, 0.5, 1.0],
        "bootstrap"         : [True, False],
        "bootstrap_features": [True, False],
      },
      "XGB"    : {
        "n_estimators"    : [10, 50, 100, 200, 500, 1000],
        "learning_rate"   : [0.01, 0.1, 0.5, 1, 10],
        "max_depth"       : [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5, 10],
        "subsample"       : [0.5, 0.7, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "gamma"           : [0, 0.1, 0.5, 1],
        "reg_alpha"       : [0, 0.1, 1],
        "reg_lambda"      : [0, 0.1, 1],
      },
    }


if __name__ == "__main__":
  configs = Configs()
  print(configs.modelsHyperparameters)
  # Example: Accessing hyperparameters for Random Forest
  rfHyperparams = configs.modelsHyperparameters["RF"]
  print("Random Forest Hyperparameters:", rfHyperparams)
