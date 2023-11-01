import numpy as np


def stochastic_universal_sampling(population, fitness_values, num_parents):
    """
    Perform Stochastic Universal Sampling (SUS) to select parents from the population.

    Args:
        population (numpy.ndarray): The population to select from.
        fitness_values (numpy.ndarray): An array of fitness values for each individual.
        num_parents (int): The number of parents to select.

    Returns:
        numpy.ndarray: An array of selected parents from the population.
    """
    if not isinstance(population, np.ndarray) or not isinstance(fitness_values, np.ndarray):
        raise ValueError("Inputs 'population' and 'fitness_values' must be numpy arrays.")
    if num_parents <= 0 or num_parents > len(population):
        raise ValueError("Invalid number of parents to select.")

    total_fitness = sum(fitness_values)
    pointers = np.linspace(0, total_fitness, num_parents + 1)
    selected_parents = []

    cum_fitness = 0
    i = 0
    for p in pointers:
        while cum_fitness < p and i < len(population):
            cum_fitness += fitness_values[i]
            i += 1
        selected_parents.append(population[i - 1])

    return np.array(selected_parents)

def k_tournament_selection(population, fitness_values, num_parents, k):
    """
    Perform K-Tournament Selection to select parents from the population.

    Args:
        population (numpy.ndarray): The population to select from.
        fitness_values (numpy.ndarray): An array of fitness values for each individual.
        num_parents (int): The number of parents to select.
        k (int): The size of each tournament.

    Returns:
        numpy.ndarray: An array of selected parents from the population.
    """
    if not isinstance(population, np.ndarray) or not isinstance(fitness_values, np.ndarray):
        raise ValueError("Inputs 'population' and 'fitness_values' must be numpy arrays.")
    if num_parents <= 0 or num_parents > len(population) or k <= 0:
        raise ValueError("Invalid number of parents or tournament size.")

    selected_parents = []
    for _ in range(num_parents):
        tournament_indices = np.random.choice(len(population), k, replace=False)
        tournament_fitness = fitness_values[tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        selected_parents.append(population[winner_index])

    return np.array(selected_parents)

def rank_based_selection(population, fitness_values, num_parents):
    """
    Perform Rank-Based Selection to select parents from the population.

    Args:
        population (numpy.ndarray): The population to select from.
        fitness_values (numpy.ndarray): An array of fitness values for each individual.
        num_parents (int): The number of parents to select.

    Returns:
        numpy.ndarray: An array of selected parents from the population.
    """
    if not isinstance(population, np.ndarray) or not isinstance(fitness_values, np.ndarray):
        raise ValueError("Inputs 'population' and 'fitness_values' must be numpy arrays.")
    if num_parents <= 0 or num_parents > len(population):
        raise ValueError("Invalid number of parents to select.")

    ranking = np.argsort(fitness_values)
    selection_probabilities = (2 * (ranking + 1)) / (len(population) * (len(population) + 1))
    selected_indices = np.random.choice(len(population), num_parents, p=selection_probabilities)
    selected_parents = population[selected_indices]

    return selected_parents

def exponential_selection(population, fitness_values, num_parents, beta):
    """
    Perform Exponential Selection to select parents from the population.

    Args:
        population (numpy.ndarray): The population to select from.
        fitness_values (numpy.ndarray): An array of fitness values for each individual.
        num_parents (int): The number of parents to select.
        beta (float): The selection pressure parameter.

    Returns:
        numpy.ndarray: An array of selected parents from the population.
    """
    if not isinstance(population, np.ndarray) or not isinstance(fitness_values, np.ndarray):
        raise ValueError("Inputs 'population' and 'fitness_values' must be numpy arrays.")
    if num_parents <= 0 or num_parents > len(population) or beta <= 0:
        raise ValueError("Invalid number of parents or selection pressure parameter.")

    # Normalize fitness values to probabilities using softmax
    max_fit = np.max(fitness_values)
    exp_values = np.exp(beta * (fitness_values - max_fit))
    selection_probs = exp_values / np.sum(exp_values)
    
    # Perform selection with calculated probabilities
    selected_indices = np.random.choice(len(population), num_parents, p=selection_probs)
    selected_parents = population[selected_indices]

    return selected_parents


# Number of parents to select
num_parents = 3

# Test Stochastic Universal Sampling
selected_parents_sus = stochastic_universal_sampling(initial_population, fitness_values, num_parents)
print("Stochastic Universal Sampling:")
print(selected_parents_sus)

# Test K-Tournament Selection
k = 2  # Tournament size
selected_parents_kt = k_tournament_selection(initial_population, fitness_values, num_parents, k)
print("\nK-Tournament Selection:")
print(selected_parents_kt)

# Test Rank-Based Selection
selected_parents_rank = rank_based_selection(initial_population, fitness_values, num_parents)
print("\nRank-Based Selection:")
print(selected_parents_rank)

# Test Exponential Selection
beta = 0.2  # Selection pressure parameter
selected_parents_exp = exponential_selection(initial_population, fitness_values, num_parents, beta)
print("\nExponential Selection:")
print(selected_parents_exp)