# Import necessary libraries.
import IgnoreWarnings
import tqdm  # Import the tqdm library for progress bars.
import numpy as np  # Import NumPy for numerical operations.


class ParticleSwarmOptimization(object):
  """
  A class implementing the Particle Swarm Optimization (PSO) algorithm.

  Particle Swarm Optimization is a population-based optimization technique inspired by the social behavior of birds flocking or fish schooling.
  This implementation supports both maximization and minimization problems.

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
      The population size, i.e., the number of particles in the swarm.

  T : int
      The number of iterations (generations) for which the algorithm will run.

  w : float, optional
      The inertia weight, controlling the influence of the previous velocity on the current velocity. Default is 0.5.

  c1 : float, optional
      The cognitive parameter, influencing the particle's tendency to move toward its personal best position. Default is 1.5.

  c2 : float, optional
      The social parameter, influencing the particle's tendency to move toward the global best position. Default is 1.5.

  isMax : bool, optional
      A boolean indicating whether to maximize (True) or minimize (False) the objective function. Default is True.

  Attributes:
  ----------
  pop : numpy.ndarray
      The current positions of all particles in the swarm.

  velocities : numpy.ndarray
      The velocities of all particles in the swarm.

  fitness : numpy.ndarray
      The fitness values of all particles in the swarm.

  personalBestPositions : numpy.ndarray
      The best positions each particle has encountered so far.

  personalBestFitness : numpy.ndarray
      The fitness values corresponding to the personal best positions.

  bestX : numpy.ndarray
      The global best position found by the swarm.

  bestF : float
      The fitness value corresponding to the global best position.

  histBestFit : numpy.ndarray
      A history of the best fitness values over all iterations.

  Methods:
  -------
  EvaluateFitness(x)
      Evaluate the fitness of a single particle.

  EvaluateFitnessAll()
      Evaluate the fitness of all particles in the population.

  UpdatePersonalBest()
      Update the personal best positions and fitness values.

  UpdateGlobalBest()
      Update the global best position and fitness value.

  SpaceBound(x)
      Ensure that a particle's position stays within the specified bounds.

  Optimize()
      Run the PSO optimization process.
  """

  def __init__(self, objectiveFunction, lb, ub, D, ps, T, w=0.5, c1=1.5, c2=1.5, isMax=True):
    """
    Initialize the Particle Swarm Optimization (PSO) algorithm.

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
        Population size (number of particles).

    T : int
        Number of iterations.

    w : float, optional
        Inertia weight (default: 0.5).

    c1 : float, optional
        Cognitive parameter (default: 1.5).

    c2 : float, optional
        Social parameter (default: 1.5).

    isMax : bool, optional
        Boolean indicating whether to maximize (True) or minimize (False).
    """
    self.objectiveFunction = objectiveFunction
    self.lb = lb
    self.ub = ub
    self.D = D
    self.ps = ps
    self.T = T
    self.w = w
    self.wMax = 0.9
    self.wMin = 0.4
    self.c1 = c1
    self.c2 = c2
    self.isMax = isMax

    # Initialize particle positions, velocities, and fitness values.
    self.pop = np.random.uniform(self.lb, self.ub, (self.ps, self.D))
    self.velocities = np.zeros((self.ps, self.D))
    self.fitness = np.zeros(self.ps)

    # Initialize personal best positions and fitness values.
    self.personalBestPositions = np.copy(self.pop)
    self.personalBestFitness = np.full(self.ps, -float("inf") if isMax else float("inf"))

    # Initialize global best position and fitness value.
    self.bestX = np.zeros(self.D)
    self.bestF = -float("inf") if isMax else float("inf")

    # History of the best fitness values over iterations.
    self.histBestFit = np.zeros(self.T)

  def EvaluateFitness(self, x):
    """
    Evaluate the fitness of a single particle.

    Parameters:
    ----------
    x : numpy.ndarray
        The position of the particle.

    Returns:
    -------
    float
        The fitness value of the particle.
    """
    return self.objectiveFunction(x.tolist())

  def EvaluateFitnessAll(self):
    """
    Evaluate the fitness of all particles in the population.

    This method updates the `fitness` attribute with the fitness values of all particles.
    """
    for i in range(self.ps):
      self.fitness[i] = self.EvaluateFitness(self.pop[i])

  def UpdatePersonalBest(self):
    """
    Update the personal best positions and fitness values.

    Each particle updates its personal best position if its current fitness is better.
    """
    for i in range(self.ps):
      if (self.isMax):
        if (self.fitness[i] > self.personalBestFitness[i]):
          self.personalBestFitness[i] = self.fitness[i]
          self.personalBestPositions[i] = self.pop[i]
      else:
        if (self.fitness[i] < self.personalBestFitness[i]):
          self.personalBestFitness[i] = self.fitness[i]
          self.personalBestPositions[i] = self.pop[i]

  def UpdateGlobalBest(self):
    """
    Update the global best position and fitness value.

    The global best position is updated based on the best personal best fitness values.
    """
    if (self.isMax):
      idx = np.argmax(self.personalBestFitness)
    else:
      idx = np.argmin(self.personalBestFitness)

    if (self.isMax):
      if (self.personalBestFitness[idx] > self.bestF):
        self.bestF = self.personalBestFitness[idx]
        self.bestX = self.personalBestPositions[idx]
    else:
      if (self.personalBestFitness[idx] < self.bestF):
        self.bestF = self.personalBestFitness[idx]
        self.bestX = self.personalBestPositions[idx]

  def SpaceBound(self, x):
    """
    Ensure the particle stays within the specified bounds.

    Parameters:
    ----------
    x : numpy.ndarray
        The position of the particle.

    Returns:
    -------
    numpy.ndarray
        The bounded position of the particle.
    """
    return np.clip(x, self.lb, self.ub)

  def Optimize(self):
    """
    Run the PSO optimization process.

    This method iteratively updates particle positions and velocities, evaluates fitness,
    and updates personal and global bests until the maximum number of iterations is reached.
    """
    # Initial evaluation of fitness and updates.
    self.EvaluateFitnessAll()
    self.UpdatePersonalBest()
    self.UpdateGlobalBest()

    # Create a progress bar for iterations.
    itLoop = tqdm.tqdm(range(1, self.T + 1), desc="Iterations", leave=False)

    for it in itLoop:
      for i in range(self.ps):
        # Update velocity.
        r1, r2 = np.random.rand(self.D), np.random.rand(self.D)
        cognitive = self.c1 * r1 * (self.personalBestPositions[i] - self.pop[i])
        social = self.c2 * r2 * (self.bestX - self.pop[i])
        self.velocities[i] = self.w * self.velocities[i] + cognitive + social

        # Update position.
        self.pop[i] += self.velocities[i]
        self.pop[i] = self.SpaceBound(self.pop[i])

        # Evaluate fitness of the new position.
        self.fitness[i] = self.EvaluateFitness(self.pop[i])

      # Update inertia weight.
      self.w = self.wMax - ((self.wMax - self.wMin) * it / self.T)

      # Update personal and global bests.
      self.UpdatePersonalBest()
      self.UpdateGlobalBest()

      # Record the best fitness value for this iteration.
      self.histBestFit[it - 1] = self.bestF

      # Log the best fitness value in the progress bar.
      itLoop.set_postfix({"Best Fitness": self.bestF})

    # Optionally return the best solution, fitness, and history.
    # return self.bestX, self.bestF, self.histBestFit
