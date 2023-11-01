import random
import numpy as np
from numba import jit

class TSPPopulationMutator:
    def __init__(self, population, mutation_rate=0.1):
        """
        Initialize the TSPPopulationMutator.

        Parameters:
        - population: Initial population as a NumPy array.
        - mutation_rate: Probability of mutation (0.1 for 10% mutation rate).
        """
        assert isinstance(population, np.ndarray), "Population must be a NumPy array"
        assert population.dtype == int, "Population elements must be of integer type"
        self.population = population
        self.mutation_rate = mutation_rate

    @jit
    def swap_mutation(self):
        """
        Apply Swap Mutation to the population.

        Returns:
        A mutated population with swap mutations.
        """
        assert self.population.dtype == int, "Population elements must be of integer type"
        mutated_population = self.population.copy()

        for i in range(len(mutated_population)):
            if random.random() < self.mutation_rate:
                idx1, idx2 = np.random.choice(len(mutated_population[i]), 2, replace=False)
                mutated_population[i, [idx1, idx2]] = mutated_population[i, [idx2, idx1]]

        return mutated_population

    @jit
    def insert_mutation(self):
        """
        Apply Insert Mutation to the population.

        Returns:
        A mutated population with insert mutations.
        """
        assert self.population.dtype == int, "Population elements must be of integer type"
        mutated_population = self.population.copy()

        for i in range(len(mutated_population)):
            if random.random() < self.mutation_rate:
                idx1, idx2 = np.random.choice(len(mutated_population[i]), 2, replace=False)
                gene = mutated_population[i, idx1]
                mutated_population[i] = np.insert(mutated_population[i], idx2, gene)
                mutated_population[i, idx1] = mutated_population[i, idx1 + 1]

        return mutated_population

    @jit
    def scramble_mutation(self):
        """
        Apply Scramble Mutation to the population.

        Returns:
        A mutated population with scramble mutations.
        """
        assert self.population.dtype == int, "Population elements must be of integer type"
        mutated_population = self.population.copy()

        for i in range(len(mutated_population)):
            if random.random() < self.mutation_rate:
                idx1, idx2 = np.random.choice(len(mutated_population[i]), 2, replace=False)
                segment = mutated_population[i, idx1:idx2]
                np.random.shuffle(segment)
                mutated_population[i, idx1:idx2] = segment

        return mutated_population

    @jit
    def inverse_mutation(self):
        """
        Apply Inverse Mutation to the population.

        Returns:
        A mutated population with inverse mutations.
        """
        assert self.population.dtype == int, "Population elements must be of integer type"
        mutated_population = self.population.copy()

        for i in range(len(mutated_population)):
            if random.random() < self.mutation_rate:
                idx1, idx2 = np.random.choice(len(mutated_population[i]), 2, replace=False)
                mutated_population[i, idx1:idx2] = np.flip(mutated_population[i, idx1:idx2])

        return mutated_population

# Example usage
mutation_rate = 0.1

# Assuming you have the initial population created earlier (heuristic_population)
mutator = TSPPopulationMutator(heuristic_population, mutation_rate)

# Apply swap mutation to the population
mutated_population = mutator.swap_mutation()

# Apply insert mutation to the population
mutated_population = mutator.insert_mutation()

# Apply scramble mutation to the population
mutated_population = mutator.scramble_mutation()

# Apply inverse mutation to the population
mutated_population = mutator.inverse_mutation()

