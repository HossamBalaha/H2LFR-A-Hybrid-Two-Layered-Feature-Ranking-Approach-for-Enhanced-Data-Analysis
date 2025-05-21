# Import necessary libraries.
import IgnoreWarnings
import tqdm  # Import the tqdm library for progress bars.
import numpy as np  # Import NumPy for numerical operations.


class MantaRayForagingAlgorithm(object):
  """
  A class implementing the Manta Ray Foraging Optimization (MRFO) algorithm.

  The MRFO algorithm is a nature-inspired optimization technique that mimics the foraging behavior of manta rays.
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
      The population size, i.e., the number of solutions (manta rays) in the swarm.

  T : int
      The number of iterations (generations) for which the algorithm will run.

  isMax : bool, optional
      A boolean indicating whether to maximize (True) or minimize (False) the objective function. Default is True.

  Attributes:
  ----------
  pop : numpy.ndarray
      The current population of solutions (manta rays).

  fitness : numpy.ndarray
      The fitness values of all solutions in the population.

  bestX : numpy.ndarray
      The best solution found by the algorithm.

  bestF : float
      The fitness value corresponding to the best solution.

  histBestFit : numpy.ndarray
      A history of the best fitness values over all iterations.

  Methods:
  -------
  InitializePopulation()
      Initialize the population with random values within the specified bounds.

  EvaluateFitness(x)
      Evaluate the fitness of a single solution.

  EvaluateFitnessAll()
      Evaluate the fitness of all solutions in the population.

  UpdateBest()
      Update the best solution and its fitness value.

  SpaceBound(x)
      Ensure that a solution stays within the specified bounds.

  Optimize()
      Run the MRFO optimization process.
  """

  def __init__(self, objectiveFunction, lb, ub, D, ps, T, isMax=True):
    """
    Initialize the Manta Ray Foraging Optimization (MRFO) algorithm.

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
        Population size (number of solutions).

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

    # Initialize the population matrix with zeros.
    self.pop = np.zeros((self.ps, self.D))

    # Initialize the fitness array with zeros.
    self.fitness = np.zeros(self.ps)

    # Initialize the best solution vector with zeros.
    self.bestX = np.zeros(self.D)

    # Initialize an array to store the history of the best fitness values.
    self.histBestFit = np.zeros(self.T)

    # Set the initial best fitness value based on maximization or minimization.
    if (self.isMax):
      self.bestF = -float("inf")  # Best fitness for maximization.
    else:
      self.bestF = float("inf")  # Best fitness for minimization.

    # Initialize the population with random values within the bounds.
    self.InitializePopulation()

  def InitializePopulation(self):
    """
    Initialize the population with random values within the specified bounds.

    This method generates a random population of solutions and ensures precision by storing it as float64.
    """
    pop = np.random.uniform(self.lb, self.ub, (self.ps, self.D))
    pop = pop.astype(np.float64)
    self.pop = pop

  def EvaluateFitness(self, x):
    """
    Evaluate the fitness of a single solution using the objective function.

    Parameters:
    ----------
    x : numpy.ndarray
        The solution to evaluate.

    Returns:
    -------
    float
        The fitness value of the solution.
    """
    result = self.objectiveFunction(x.copy().tolist())
    return result

  def EvaluateFitnessAll(self):
    """
    Evaluate the fitness of all solutions in the population.

    This method updates the `fitness` attribute with the fitness values of all solutions.
    """
    fitness = np.zeros(self.ps)
    for i in range(self.ps):
      fitness[i] = self.EvaluateFitness(self.pop[i, :])
    self.fitness = fitness

  def UpdateBest(self):
    """
    Update the best solution and its fitness value.

    This method identifies the best solution in the current population and updates the global best.
    """
    if (self.isMax):
      idx = np.argmax(self.fitness)
    else:
      idx = np.argmin(self.fitness)

    self.bestX = self.pop[idx, :].astype(np.float64)
    self.bestF = self.fitness[idx].astype(np.float64)

  def SpaceBound(self, x):
    """
    Ensure that a solution stays within the specified bounds.

    Parameters:
    ----------
    x : numpy.ndarray
        The solution to clip.

    Returns:
    -------
    numpy.ndarray
        The bounded solution.
    """
    x = np.clip(x, self.lb, self.ub)
    return x

  def Optimize(self):
    """
    Run the MRFO optimization process.

    This method iteratively applies the MRFO algorithm's movement strategies to update the population,
    evaluates fitness, and tracks the best solution over the specified number of iterations.
    """
    # Evaluate the fitness of all individuals in the initial population.
    self.EvaluateFitnessAll()

    # Update the best solution and fitness value.
    self.UpdateBest()

    # Create a progress bar for the iterations.
    itLoop = tqdm.tqdm(range(1, self.T + 1), desc="Iterations", leave=False)

    # Iterate through the specified number of iterations.
    for it in itLoop:
      # Calculate the coefficient for the current iteration.
      coef = float(it) / float(self.T)

      # Initialize a new population matrix.
      newPop = np.zeros_like(self.pop).astype(np.float64)

      # Perform the first phase of the MRFO algorithm.
      for i in range(self.ps):
        # Check if the first movement strategy should be applied.
        if (np.random.rand() < 0.5):
          # Generate a random value for beta calculation.
          r1 = np.random.rand()
          beta = 2.0 * np.exp(r1 * ((self.T - it + 1) / self.T)) * np.sin(2 * np.pi * r1)
          beta = beta.astype(np.float64)

          # Apply the first movement strategy based on the coefficient.
          if (coef > np.random.rand()):
            newPop[i, :] = (
              self.bestX
              + np.random.rand(self.D) * (self.bestX - self.pop[i, :])
              + beta * (self.bestX - self.pop[i, :])
            )
          else:
            indivRand = np.random.rand(self.D) * (self.ub - self.lb) + self.lb
            newPop[i, :] = (
              indivRand
              + np.random.rand(self.D) * (indivRand - self.pop[i, :])
              + beta * (indivRand - self.pop[i, :])
            )
        else:
          # Apply the second movement strategy.
          alpha = 2.0 * np.random.rand(self.D) * (-np.log(np.random.rand(self.D))) ** 0.5
          newPop[i, :] = (
            self.pop[i, :]
            + np.random.rand(self.D) * (self.bestX - self.pop[i, :])
            + alpha * (self.bestX - self.pop[i, :])
          )

        # Ensure the new solution stays within the bounds.
        newPop[i, :] = self.SpaceBound(newPop[i, :]).astype(np.float64)

        # Evaluate the fitness of the new solution.
        newFit = self.EvaluateFitness(newPop[i, :])

        # Update the population and fitness if the new solution is better.
        if (self.isMax):
          if (newFit > self.fitness[i]):
            self.fitness[i] = newFit
            self.pop[i, :] = newPop[i, :].astype(np.float64)
        else:
          if (newFit < self.fitness[i]):
            self.fitness[i] = newFit
            self.pop[i, :] = newPop[i, :].astype(np.float64)

      # Perform the second phase of the MRFO algorithm.
      for i in range(self.ps):
        # Apply the third movement strategy.
        newPop[i, :] = (
          self.pop[i, :]
          + 2.0 * (np.random.rand() * self.bestX - np.random.rand() * self.pop[i, :])
        )
        newPop[i, :] = self.SpaceBound(newPop[i, :]).astype(np.float64)

        # Evaluate the fitness of the new solution.
        newFit = self.EvaluateFitness(newPop[i, :])

        # Update the population and fitness if the new solution is better.
        if (self.isMax):
          if (newFit > self.fitness[i]):
            self.fitness[i] = newFit
            self.pop[i, :] = newPop[i, :].astype(np.float64)
        else:
          if (newFit < self.fitness[i]):
            self.fitness[i] = newFit
            self.pop[i, :] = newPop[i, :].astype(np.float64)

      # Update the best solution and fitness value.
      self.UpdateBest()

      # Store the best fitness value for the current iteration.
      self.histBestFit[it - 1] = self.bestF

      # Log the best fitness value in the progress bar.
      itLoop.set_postfix({"Best Fitness": self.bestF})

    # Optionally return the best solution, fitness, and history.
    # return self.bestX, self.bestF, self.histBestFit
