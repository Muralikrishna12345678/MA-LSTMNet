import time
import numpy as np
from scipy.spatial.distance import cdist

def EOO(pop_position, func, lb, ub, max_iter):
    # Initialize the population
    n_pop, dim = pop_position.shape[0], pop_position.shape[1]
    pop_fitness = np.zeros(n_pop)
    # Update the leader position
    leader_position = pop_position[0]
    convergence_curve = np.zeros(max_iter)
    # Main loop
    ct = time.time()
    for i in range(max_iter):
        # Evaluate the fitness of each individual
        for j in range(n_pop):
            pop_fitness[j] = func(pop_position[j])
        # Sort the population by fitness
        sorted_indices = np.argsort(pop_fitness)
        pop_fitness = pop_fitness[sorted_indices]
        # Update the position of each individual
        for j in range(1, n_pop):
            # Compute the distance to the leader
            d = cdist(pop_position[j].reshape(1, -1), leader_position.reshape(1, -1))[0, 0]
            # Compute the displacement vector
            delta = (pop_position[j] - leader_position) / d
            # Update the position of the individual
            pop_position[j] = pop_position[j] + delta * np.random.normal(0, 1, dim)
            # Enforce the bounds
            pop_position[j] = np.clip(pop_position[j], lb[j], ub[j])
        convergence_curve[i] = np.min(pop_fitness)
    # Evaluate the fitness of each individual one last time
    for j in range(n_pop):
        pop_fitness[j] = func(pop_position[j])

    ct = time.time() - ct
    best_solution = leader_position
    best_fitness = np.min(pop_position)
    return best_fitness, convergence_curve, best_solution, ct

