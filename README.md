# H2LFR: A Hybrid Two-Layered Feature Ranking Approach for Enhanced Data Analysis

The framework reflects the implementation of the study titled
**"H2LFR: A Hybrid Two-Layered Feature Ranking Approach for Enhanced Data Analysis"**,
authored by Balaha et al., and published in *Knowledge and Information Systems* (Springer).
DOI: [10.1007/s10115-025-02463-w](https://doi.org/10.1007/s10115-025-02463-w).

## Authors

- **Hossam Magdy Balaha**
    - Bioengineering Department, J.B. Speed School of Engineering, University of Louisville, Louisville 40292, USA
    - Computer Science and Systems Department, Faculty of Engineering, Mansoura University, Mansoura 35516, Egypt
- **Asmaa El-Sayed Hassan**
    - Mathematics and Engineering Physics Department, Faculty of Engineering, Mansoura University, Mansoura 35516, Egypt
- **Magdy Hassan Balaha**
    - Department of Obstetrics and Gynecology, Faculty of Medicine, Tanta University, Tanta 31527, Egypt

## Abstract

The study introduces **H2LFR**, a novel hybrid two-layered feature ranking (FR) technique that combines metaheuristic
optimization and weighted metrics. The approach consists of two layers:

1. **H2LFR-L1**: Focuses on extracting encoded solutions and their corresponding performance weights using metaheuristic
   optimization
   and machine learning techniques.
2. **H2LFR-L2**: Determines feature rankings based on the weights and additional calculated factors.

The framework was evaluated across diverse datasets, including medical diagnostics, biological classification, and
predictive analytics, demonstrating robustness and generalizability. H2LFR consistently outperformed or matched 16
well-known FR methods, showcasing its ability to efficiently identify relevant features and optimize predictive
performance.

## Key Contributions

1. **Hybrid Two-Layered FR Method**: Combines metaheuristics and learning algorithms for enhanced feature selection.
2. **Mathematical Steps for FR**: Introduces an 8-step process for ranking features based on encoded solutions and
   performance metrics.
3. **Performance Evaluation**: Compares the proposed approach with 16 FR methods and 8 benchmark datasets.

## Features of the Framework

- **Support for Multiple Metaheuristic Algorithms**: Includes implementations of Grey Wolf Optimizer (GWO), Manta Ray
  Foraging Algorithm (MRFO), Particle Swarm Optimization (PSO), and more.
- **Flexible Configuration**: Easily configure experiments via command-line arguments, allowing customization of
  optimization parameters, dataset handling, and model selection.
- **Dataset Preprocessing**: Built-in support for dataset splitting, normalization, encoding, and advanced sampling
  techniques (e.g., SMOTE).
- **Comprehensive Logging**: Automatically logs optimization history, convergence curves, and detailed results in
  timestamped directories for reproducibility.
- **Extensible Design**: Modular structure allows seamless integration of new algorithms, datasets, and preprocessing
  techniques.

## Project Structure

The project is organized into the following key components:

- `main.py`: Main entry point for running experiments.
- `Configs.py`: Centralized configuration management for hyperparameters and settings.
- `FeatureSelectionMeta.py`: Core logic for feature selection using metaheuristic algorithms.
- `GreyWolfOptimizer.py`, `MantaRayForagingAlgorithm.py`, `ParticleSwarmOptimization.py`: Implementations of
  metaheuristic optimization
  algorithms.
- `PerformanceMetrics.py`: Evaluation metrics for assessing model performance.
- `Helpers.py`: Utility functions for data processing, logging, and visualization.
- `IgnoreWarnings.py`: Suppresses warnings for cleaner output during execution.

## Getting Started

### Prerequisites

- **Python 3.10 or higher**
- **Required Python packages:**
    - `numpy`: For numerical operations and array handling.
    - `pandas`: For data manipulation and analysis.
    - `scikit-learn`: For machine learning models, preprocessing, and metrics.
    - `matplotlib`: For plotting convergence curves and results.
    - `imbalanced-learn`: For advanced sampling techniques (e.g., SMOTE).
    - `xgboost`: For XGBoost model support.
    - `tqdm`: For progress bar visualization during optimization.
    - `shutup`: For suppress warnings during execution.


- Tested on Windows 11, Python 3.10.16, and the following package versions:
    - `numpy`: 1.26.4
    - `pandas`: 2.2.3
    - `scikit-learn`: 1.6.1
    - `matplotlib`: 3.10.1
    - `imbalanced-learn`: 0.13.0
    - `xgboost`: 3.0.0
    - `tqdm`: 4.67.1
    - `shutup`: 0.2.0

You can install the main dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib imbalanced-learn xgboost tqdm shutup
```

### Command-Line Arguments

The following command-line arguments are supported:

- `--dataset` (str, required): Path to the dataset CSV file.
- `--storage` (str, optional): Directory to store the results. If not provided, a default directory is created based on
  the dataset name.
- `--doSplit` (bool, optional, default=True): Whether to split the dataset into training and testing sets.
- `--forceSplit` (bool, optional, default=True): Whether to force a new train-test split even if splits exist.
- `--popSize` (int, optional, default=50): Population size for the metaheuristic optimizer.
- `--numIter` (int, optional, default=50): Number of iterations for the optimizer.
- `--isMax` (bool, optional, default=True): Whether to maximize (True) or minimize (False) the fitness function.
- `--testSize` (float, optional, default=0.2): Proportion of the dataset to include in the test split.
- `--fillValue` (float, optional, default=0.0): Value to fill missing data.
- `--normTechnique` (str, optional): Normalization technique to apply (e.g., MinMaxScaler, Normalize_L2_1, etc.).
    - "Normalize_L1_0": lambda x: normalize(x, norm="l1", axis=0)
    - "Normalize_L2_0": lambda x: normalize(x, norm="l2", axis=0)
    - "Normalize_Max_0": lambda x: normalize(x, norm="max", axis=0)
    - "Normalize_L1_1": lambda x: normalize(x, norm="l1", axis=1)
    - "Normalize_L2_1": lambda x: normalize(x, norm="l2", axis=1)
    - "Normalize_Max_1": lambda x: normalize(x, norm="max", axis=1)
    - "StandardScaler": Standardize features by removing the mean and scaling to unit variance.
    - "MinMaxScaler": Transform features by scaling each feature to a given range (default is [0, 1]).
    - "MaxAbsScaler": Scale each feature by its maximum absolute value.
    - "RobustScaler": Scale features using statistics that are robust to outliers (median and IQR).
- `--sampleTechnique` (str, optional): Sampling technique to apply (e.g., SMOTE, RandomOverSampler, etc.).
    - "SMOTE": Adaptive Synthetic Sampling
    - "ADASYN": Adaptive Synthetic Sampling
    - "RandomOverSampler": Apply random oversampling
    - "RandomUnderSampler": Apply random undersampling
    - "NearMiss": NearMiss sampling
    - "TomekLinks": An undersampling method that removes Tomek links
    - "SMOTETomek": Combination of SMOTE and Tomek links
    - "SMOTEENN": Combination of SMOTE and Edited Nearest Neighbors
    - "SVMSMOTE": SVM-based SMOTE
    - "KMeansSMOTE": KMeans-based SMOTE
    - "BorderlineSMOTE": Borderline SMOTE
- `--loggingPeriod` (int, optional, default=50): Period (in iterations) for logging optimization history.
- `--top` (int, optional, default=15): Number of top features to select.
- `--meta` (str, optional, default="MRFO"): Metaheuristic algorithm to use (e.g., MRFO, GWO, PSO).
- `--modelsStrList` (str, optional, default="'DT','KNN'"): Comma-separated list of model names to use (e.g., DT,KNN,RF).
    - "RF": Random Forest Classifier
    - "DT": Decision Tree Classifier
    - "SVC": Support Vector Classifier
    - "KNN": K-Nearest Neighbors Classifier
    - "MLP": MultiLayer Perceptron Classifier
    - "AB": Adaptive Boosting Classifier
    - "GB": Gradient Boosting Classifier
    - "ETs": Extra Trees Classifier
    - "Bagging": Bagging Classifier
    - "XGB": eXtreme Gradient Boosting Classifier
- `--storeDetailedFiles` (bool, optional, default=False): Whether to store detailed result files.

## Usage Examples

Example 1 (with default parameters):
To run the framework with the default parameters, you can use the following command:

```bash
python main.py --dataset "Breast Cancer Wisconsin (Diagnostic) Dataset/Data.csv"
```

Example 2:
If you want to run the framework with the Breast Cancer Wisconsin (Diagnostic) Dataset, using Decision Tree and KNN
classifiers, with 10 iterations, you can use the following command:

```bash
python main.py --dataset "Breast Cancer Wisconsin (Diagnostic) Dataset/Data.csv" --modelsStrList "DT,KNN" --numIter 10
```

Example 3:
If you want to run the framework with the Breast Cancer Wisconsin (Diagnostic) Dataset, using Decision Tree and KNN
classifiers, with a population size of 20, 50 iterations, MinMaxScaler for normalization, GWO as the metaheuristic
algorithm, and SMOTE for sampling, you can use the following command:

```bash
python main.py --dataset "Breast Cancer Wisconsin (Diagnostic) Dataset/Data.csv" --modelsStrList "DT,KNN" --popSize 20 --numIter 50 --normTechnique MinMaxScaler --meta GWO --sampleTechnique SMOTE
```

## Results

All results are automatically stored in a timestamped directory within the specified storage location.
The directory includes:

- **Logs**: Detailed logs of the optimization process.
- **Convergence Curves**: Plots showing the progress of the optimization algorithm.
- **Selected Features**: Top features identified during the optimization process.
- **Performance Metrics**: Evaluation metrics for the selected features and models.

The framework generates convergence curves and other visualizations to help analyze the optimization process.
These are saved in both `.png` and `.pdf` formats for easy sharing and inclusion in reports.

## Citation

If you use this framework in your research, please cite the following paper:

```bibtex
@article{balaha2025h2lfr,
  title={H2LFR: a hybrid two-layered feature ranking approach for enhanced data analysis},
  author={Balaha, Hossam Magdy and Hassan, Asmaa El-Sayed and Balaha, Magdy Hassan},
  journal={Knowledge and Information Systems},
  pages={1--53},
  year={2025},
  publisher={Springer}
}
```

**DOI**: [10.1007/s10115-025-02463-w](https://doi.org/10.1007/s10115-025-02463-w)

## Copyright and License

All rights reserved. No portion of this series may be reproduced, distributed, or transmitted in any form or by any
means, such as photocopying, recording, or other electronic or mechanical methods, without the express written consent
of the author. Exceptions are made for brief quotations included in critical reviews and certain other noncommercial
uses allowed under copyright law. For inquiries regarding permission, please contact the author directly.
