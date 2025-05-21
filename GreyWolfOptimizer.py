# Import necessary libraries.
import IgnoreWarnings
import tqdm  # Import the tqdm library for progress bars.
import numpy as np  # Import NumPy for numerical operations.


class GreyWolfOptimizer(object):
  """
  A class implementing the Grey Wolf Optimizer (GWO) algorithm.

  The Grey Wolf Optimizer is a nature-inspired metaheuristic algorithm that simulates the hunting behavior of grey wolves.
  It is particularly effective for solving complex optimization problems in continuous search spaces.

  Parameters:
  ----------
  objectiveFunction : callable
      The function to be optimized. It should accept a list or array of variables and return a scalar fitness value.

  lb : float
      The lower bound for the search space. All variables will be constrained to be greater than or equal to this value.

  ub : float
      The upper bound for the search space. All variables will be constrained to be less than or equal to this value.

  D : int
      The dimension of the problem, i.e., the number of variables in the optimization problem.

  ps : int
      The population size, i.e., the number of wolves in the pack.

  T : int
      The number of iterations (generations) for which the algorithm will run.

  isMax : bool, optional
      A boolean indicating whether to maximize (True) or minimize (False) the objective function. Default is True.

  Attributes:
  ----------
  pop : numpy.ndarray
      The current positions of all wolves in the population.

  fitness : numpy.ndarray
      The fitness values of all wolves in the population.

  bestX : numpy.ndarray
      The position of the alpha wolf (best solution found).

  betaPosition : numpy.ndarray
      The position of the beta wolf (second-best solution found).

  deltaPosition : numpy.ndarray
      The position of the delta wolf (third-best solution found).

  bestF : float
      The fitness value corresponding to the alpha wolf.

  betaFitness : float
      The fitness value corresponding to the beta wolf.

  deltaFitness : float
      The fitness value corresponding to the delta wolf.

  histBestFit : numpy.ndarray
      A history of the best fitness values over all iterations.

  Methods:
  -------
  EvaluateFitness(x)
      Evaluate the fitness of a single wolf.

  EvaluateFitnessAll()
      Evaluate the fitness of all wolves in the population.

  UpdateAlphaBetaDelta()
      Update the positions of the alpha, beta, and delta wolves based on fitness values.

  SpaceBound(x)
      Ensure that a wolf's position stays within the specified bounds.

  Optimize()
      Run the Grey Wolf Optimization process.
  """

  def __init__(self, objectiveFunction, lb, ub, D, ps, T, isMax=True):
    """
    Initialize the Grey Wolf Optimizer (GWO) algorithm.

    Parameters:
    ----------
    objectiveFunction : callable
        The function to be optimized.

    lb : float
        Lower bound for the search space.

    ub : float
        Upper bound for the search space.

    D : int
        Dimension of the problem (number of variables).

    ps : int
        Population size (number of wolves).

    T : int
        Number of iterations.

    isMax : bool, optional
        Boolean indicating whether to maximize (True) or minimize (False).
    """
    self.objectiveFunction = objectiveFunction
    self.lb = lb
    self.ub = ub
    self.D = D
    self.ps = ps
    self.T = T
    self.isMax = isMax

    # Initialize wolf positions and fitness values.
    self.pop = np.random.uniform(self.lb, self.ub, (self.ps, self.D))
    self.fitness = np.zeros(self.ps)

    # Initialize alpha, beta, and delta wolves.
    self.bestX = np.zeros(self.D)
    self.betaPosition = np.zeros(self.D)
    self.deltaPosition = np.zeros(self.D)

    # Initialize fitness values for alpha, beta, and delta wolves.
    self.bestF = -float("inf") if isMax else float("inf")
    self.betaFitness = -float("inf") if isMax else float("inf")
    self.deltaFitness = -float("inf") if isMax else float("inf")

    # History of the best fitness values over iterations.
    self.histBestFit = np.zeros(self.T)

  def EvaluateFitness(self, x):
    """
    Evaluate the fitness of a single wolf.

    Parameters:
    ----------
    x : numpy.ndarray
        The position of the wolf.

    Returns:
    -------
    float
        The fitness value of the wolf.
    """
    return self.objectiveFunction(x.tolist())

  def EvaluateFitnessAll(self):
    """
    Evaluate the fitness of all wolves in the population.

    This method updates the `fitness` attribute with the fitness values of all wolves.
    """
    for i in range(self.ps):
      self.fitness[i] = self.EvaluateFitness(self.pop[i])

  def UpdateAlphaBetaDelta(self):
    """
    Update the alpha, beta, and delta wolves based on fitness values.

    This method identifies the top three solutions in the population and assigns them to the alpha, beta, and delta wolves.
    """
    # Sort the population indices based on fitness values.
    sortedIndices = np.argsort(self.fitness)[::-1] if self.isMax else np.argsort(self.fitness)

    # Update the alpha wolf (best solution).
    potentialWinner = self.fitness[sortedIndices[0]]
    if (self.isMax):
      if (potentialWinner > self.bestF):
        self.bestX = self.pop[sortedIndices[0]].copy()
        self.bestF = potentialWinner
    else:
      if (potentialWinner < self.bestF):
        self.bestX = self.pop[sortedIndices[0]].copy()
        self.bestF = potentialWinner

    # Update the beta wolf (second-best solution).
    self.betaPosition = self.pop[sortedIndices[1]].copy()
    self.betaFitness = self.fitness[sortedIndices[1]]

    # Update the delta wolf (third-best solution).
    self.deltaPosition = self.pop[sortedIndices[2]].copy()
    self.deltaFitness = self.fitness[sortedIndices[2]]

  def SpaceBound(self, x):
    """
    Ensure that a wolf's position stays within the specified bounds.

    Parameters:
    ----------
    x : numpy.ndarray
        The position of the wolf.

    Returns:
    -------
    numpy.ndarray
        The bounded position of the wolf.
    """
    return np.clip(x, self.lb, self.ub)

  def Optimize(self):
    """
    Run the Grey Wolf Optimization process.

    This method iteratively updates the positions of the wolves, evaluates fitness,
    and tracks the best solution over the specified number of iterations.
    """
    # Evaluate the fitness of all wolves in the initial population.
    self.EvaluateFitnessAll()

    # Update the alpha, beta, and delta wolves.
    self.UpdateAlphaBetaDelta()

    # Create a progress bar for the iterations.
    itLoop = tqdm.tqdm(range(1, self.T + 1), desc="Iterations", leave=False)

    # Iterate through the specified number of iterations.
    for it in itLoop:
      # Decrease the parameter 'a' linearly from 2 to 0.
      a = 2 - it * (2 / self.T)

      for i in range(self.ps):
        # Compute coefficients A and C for each wolf.
        r1, r2 = np.random.rand(self.D), np.random.rand(self.D)
        A1, C1 = 2 * a * r1 - a, 2 * r2

        r1, r2 = np.random.rand(self.D), np.random.rand(self.D)
        A2, C2 = 2 * a * r1 - a, 2 * r2

        r1, r2 = np.random.rand(self.D), np.random.rand(self.D)
        A3, C3 = 2 * a * r1 - a, 2 * r2

        # Update the position of the current wolf based on the alpha, beta, and delta wolves.
        X1 = self.bestX - A1 * abs(C1 * self.bestX - self.pop[i])
        X2 = self.betaPosition - A2 * abs(C2 * self.betaPosition - self.pop[i])
        X3 = self.deltaPosition - A3 * abs(C3 * self.deltaPosition - self.pop[i])

        self.pop[i] = (X1 + X2 + X3) / 3
        self.pop[i] = self.SpaceBound(self.pop[i])

      # Evaluate the fitness of the new positions.
      self.EvaluateFitnessAll()

      # Update the alpha, beta, and delta wolves.
      self.UpdateAlphaBetaDelta()

      # Record the best fitness value for the current iteration.
      self.histBestFit[it - 1] = self.bestF

      # Log the best fitness value in the progress bar.
      itLoop.set_postfix({"Best Fitness": self.bestF})

    # Return the best solution, fitness, and history.
    return self.bestX, self.bestF, self.histBestFit
