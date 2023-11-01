import numpy as np
import random

####################Codes from Jacob#####################################################

def invMutation(individual):
    """
    Apply Inversion Mutation to an individual in-place.

    Parameters:
    - individual: A NumPy array representing an individual.

    This function mutates the individual in-place, so it returns nothing.
    """
    if np.random.uniform() < individual.alpha:
        i = np.random.randint(0, len(individual.route) - 1)
        j = np.random.randint(i + 1, len(individual.route))
        individual.route[i:j] = np.flip(individual.route[i:j])
        
def swapMutation(individual):
    """
    Apply Swap Mutation to an individual in-place.

    Parameters:
    - individual: A NumPy array representing an individual.

    This function mutates the individual in-place, so it returns nothing.
    """
    if np.random.uniform() < individual.alpha:
        i = np.random.randint(0, len(individual.route))
        j = np.random.randint(0, len(individual.route))
        tmp = individual.route[i]
        individual.route[i] = individual.route[j]
        individual.route[j] = tmp

def scrambleMutation(individual):
    """
    Apply Scramble Mutation to an individual in-place.

    Parameters:
    - individual: A NumPy array representing an individual.

    This function mutates the individual in-place, so it returns nothing.
    """
    if np.random.uniform() < individual.alpha:
        i = np.random.randint(0, len(individual.route))
        j = np.random.randint(i + 1, len(individual.route))
        segment = individual.route[i:j]
        np.random.shuffle(segment)
        individual.route[i:j] = segment

def insertMutation(individual):
    """
    Apply Insert Mutation to an individual in-place.

    Parameters:
    - individual: A NumPy array representing an individual.

    This function mutates the individual in-place, so it returns nothing.
    """
    if np.random.uniform() < individual.alpha:
        i = np.random.randint(0, len(individual.route))
        j = np.random.randint(0, len(individual.route))
        gene = individual.route[i]
        individual.route = np.insert(individual.route, j, gene)
        if j <= i:
            i += 1
        individual.route = np.delete(individual.route, i)


####################Population-input#####################################################

def tsp_swap_mutation(population, mutation_rate):
    """
    Apply Swap Mutation to a population.

    Parameters:
    - population: Initial population as a NumPy array.
    - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).

    Returns:
    A mutated population with swap mutations.
    """
    assert isinstance(population, np.ndarray), "Population must be a NumPy array"
    assert population.dtype == int, "Population elements must be of integer type"

    mutated_population = population.copy()

    for i in range(len(mutated_population)):
        if random.random() < mutation_rate:
            idx1, idx2 = np.random.choice(len(mutated_population[i]), 2, replace=False)
            mutated_population[i, [idx1, idx2]] = mutated_population[i, [idx2, idx1]]

    return mutated_population

def tsp_insert_mutation(population, mutation_rate):
    """
    Apply Insert Mutation to a population.

    Parameters:
    - population: Initial population as a NumPy array.
    - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).

    Returns:
    A mutated population with insert mutations.
    """
    assert isinstance(population, np.ndarray), "Population must be a NumPy array"
    assert population.dtype == int, "Population elements must be of integer type"

    mutated_population = population.copy()

    for i in range(len(mutated_population)):
        if random.random() < mutation_rate:
            idx1, idx2 = np.random.choice(len(mutated_population[i]), 2, replace=False)
            gene = mutated_population[i, idx1]
            mutated_population[i] = np.insert(mutated_population[i], idx2, gene)
            mutated_population[i, idx1] = mutated_population[i, idx1 + 1]

    return mutated_population

def tsp_scramble_mutation(population, mutation_rate):
    """
    Apply Scramble Mutation to a population.

    Parameters:
    - population: Initial population as a NumPy array.
    - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).

    Returns:
    A mutated population with scramble mutations.
    """
    assert isinstance(population, np.ndarray), "Population must be a NumPy array"
    assert population.dtype == int, "Population elements must be of integer type"

    mutated_population = population.copy()

    for i in range(len(mutated_population)):
        if random.random() < mutation_rate:
            idx1, idx2 = np.random.choice(len(mutated_population[i]), 2, replace=False)
            segment = mutated_population[i, idx1:idx2]
            np.random.shuffle(segment)
            mutated_population[i, idx1:idx2] = segment

    return mutated_population

def tsp_inverse_mutation(population, mutation_rate):
    """
    Apply Inverse Mutation to a population.

    Parameters:
    - population: Initial population as a NumPy array.
    - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).

    Returns:
    A mutated population with inverse mutations.
    """
    assert isinstance(population, np.ndarray), "Population must be a NumPy array"
    assert population.dtype == int, "Population elements must be of integer type"

    mutated_population = population.copy()

    for i in range(len(mutated_population)):
        if random.random() < mutation_rate:
            idx1, idx2 = np.random.choice(len(mutated_population[i]), 2, replace=False)
            mutated_population[i, idx1:idx2] = np.flip(mutated_population[i, idx1:idx2])

    return mutated_population

# Example usage
mutation_rate = 0.1

# Assuming you have the initial population created earlier (heuristic_population)
mutated_population = tsp_swap_mutation(heuristic_population, mutation_rate)

# Apply insert mutation to the population
mutated_population = tsp_insert_mutation(heuristic_population, mutation_rate)

# Apply scramble mutation to the population
mutated_population = tsp_scramble_mutation(heuristic_population, mutation_rate)

# Apply inverse mutation to the population
mutated_population = tsp_inverse_mutation(heuristic_population, mutation_rate)

####################Individual-input#####################################################
import numpy as np
import random
from numba import jit

def tsp_swap_mutation(individual, mutation_rate):
    """
    Apply Swap Mutation to an individual.

    Parameters:
    - individual: A NumPy array representing an individual.
    - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).

    Returns:
    A mutated individual with swap mutations.
    """
    assert isinstance(individual, np.ndarray), "Individual must be a NumPy array"
    assert individual.dtype == int, "Individual elements must be of integer type"

    mutated_individual = individual.copy()

    if random.random() < mutation_rate:
        idx1, idx2 = np.random.choice(len(mutated_individual), 2, replace=False)
        mutated_individual[[idx1, idx2]] = mutated_individual[[idx2, idx1]]

    return mutated_individual

def tsp_insert_mutation(individual, mutation_rate):
    """
    Apply Insert Mutation to an individual.

    Parameters:
    - individual: A NumPy array representing an individual.
    - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).

    Returns:
    A mutated individual with insert mutations.
    """
    assert isinstance(individual, np.ndarray), "Individual must be a NumPy array"
    assert individual.dtype == int, "Individual elements must be of integer type"

    mutated_individual = individual.copy()

    if random.random() < mutation_rate:
        idx1, idx2 = np.random.choice(len(mutated_individual), 2, replace=False)
        gene = mutated_individual[idx1]
        mutated_individual = np.insert(mutated_individual, idx2, gene)
        mutated_individual[idx1] = mutated_individual[idx1 + 1]

    return mutated_individual

def tsp_scramble_mutation(individual, mutation_rate):
    """
    Apply Scramble Mutation to an individual.

    Parameters:
    - individual: A NumPy array representing an individual.
    - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).

    Returns:
    A mutated individual with scramble mutations.
    """
    assert isinstance(individual, np.ndarray), "Individual must be a NumPy array"
    assert individual.dtype == int, "Individual elements must be of integer type"

    mutated_individual = individual.copy()

    if random.random() < mutation_rate:
        idx1, idx2 = np.random.choice(len(mutated_individual), 2, replace=False)
        segment = mutated_individual[idx1:idx2]
        np.random.shuffle(segment)
        mutated_individual[idx1:idx2] = segment

    return mutated_individual

def tsp_inverse_mutation(individual, mutation_rate):
    """
    Apply Inverse Mutation to an individual.

    Parameters:
    - individual: A NumPy array representing an individual.
    - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).

    Returns:
    A mutated individual with inverse mutations.
    """
    assert isinstance(individual, np.ndarray), "Individual must be a NumPy array"
    assert individual.dtype == int, "Individual elements must be of integer type"

    mutated_individual = individual.copy()

    if random.random() < mutation_rate:
        idx1, idx2 = np.random.choice(len(mutated_individual), 2, replace=False)
        mutated_individual[idx1:idx2] = np.flip(mutated_individual[idx1:idx2])

    return mutated_individual

# Example usage
mutation_rate = 0.1

# Assuming you have an initial individual created earlier (heuristic_individual)
mutated_individual = tsp_swap_mutation(heuristic_individual, mutation_rate)

# Apply insert mutation to the individual
mutated_individual = tsp_insert_mutation(heuristic_individual, mutation_rate)

# Apply scramble mutation to the individual
mutated_individual = tsp_scramble_mutation(heuristic_individual, mutation_rate)

# Apply inverse mutation to the individual
mutated_individual = tsp_inverse_mutation(heuristic_individual, mutation_rate)

