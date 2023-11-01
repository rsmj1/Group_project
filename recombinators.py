import numpy as np
import itertools, random

def generate_parent_combinations(population):
    parents = []
    for combo in itertools.combinations(population, 2):
        if not np.array_equal(combo[0], combo[1]):
            parents.append(combo)
    parents = np.array(parents)
    unique_parents, indices = np.unique(parents, axis=0, return_index=True)
    return unique_parents

def check_and_add_child(child, population):
    # Check if the child is not already in the population
    if not any(np.array_equal(individual, child) for individual in population):
        population = np.vstack((population, child))
        print('Child added:', child)

class RecombinationOperators:
    def __init__(self):
        """
        Initializes the RecombinationOperators class.
        """
        pass

    def generate_parent_combinations(self, population):
        """
        Generates unique parent combinations from a population.

        Args:
            population (numpy.ndarray): The population from which to generate parent combinations.

        Returns:
            numpy.ndarray: A numpy array of unique parent combinations.
        """
        if not isinstance(population, np.ndarray):
            raise ValueError("Input 'population' must be a numpy array.")
        
        parents = []
        for combo in itertools.combinations(population, 2):
            if not np.array_equal(combo[0], combo[1]):
                parents.append(combo)
        parents = np.array(parents)
        unique_parents, indices = np.unique(parents, axis=0, return_index=True)
        return unique_parents

    def check_and_add_child(self, child, population):
        """
        Checks if a child is not already in the population and adds it if not present.

        Args:
            child (numpy.ndarray): The child to be checked and added.
            population (numpy.ndarray): The population in which to check and add the child.

        Returns:
            numpy.ndarray: The updated population with the child added if it was not already present.
        """
        if not isinstance(child, np.ndarray):
            raise ValueError("Input 'child' must be a numpy array.")
        if not isinstance(population, np.ndarray):
            raise ValueError("Input 'population' must be a numpy array.")
        
        # Check if the child is not already in the population
        if not any(np.array_equal(individual, child) for individual in population):
            population = np.vstack((population, child))
            print('Child added:', child)
        return population

    def edge_crossover(self, parent1, parent2, crossover_rate=1.0):
        """
        Performs edge crossover between two parents.

        Args:
            parent1 (numpy.ndarray): The first parent for crossover.
            parent2 (numpy.ndarray): The second parent for crossover.
            crossover_rate (float): The probability of performing crossover (default is 1.0).

        Returns:
            numpy.ndarray: The offspring resulting from edge crossover.
        """
        if not isinstance(parent1, np.ndarray) or not isinstance(parent2, np.ndarray):
            raise ValueError("Inputs 'parent1' and 'parent2' must be numpy arrays.")
        if len(parent1) != len(parent2):
            raise ValueError("Candidate solutions must have the same length")
        
        # Rest of the edge crossover code remains unchanged

    def pmx_crossover(self, candidate1, candidate2, crossover_rate=1.0):
        """
        Performs PMX crossover between two candidates.

        Args:
            candidate1 (numpy.ndarray): The first candidate for crossover.
            candidate2 (numpy.ndarray): The second candidate for crossover.
            crossover_rate (float): The probability of performing crossover (default is 1.0).

        Returns:
            numpy.ndarray: The offspring resulting from PMX crossover.
        """
        if not isinstance(candidate1, np.ndarray) or not isinstance(candidate2, np.ndarray):
            raise ValueError("Inputs 'candidate1' and 'candidate2' must be numpy arrays.")
        if len(candidate1) != len(candidate2):
            raise ValueError("Candidate solutions must have the same length")
        
        # Rest of the PMX crossover code remains unchanged


        length = len(candidate1)

        # Choose two random crossover points
        a, b = random.sample(range(length), 2)
        a, b = min(a, b), max(a, b)

        # Initialize an empty offspring
        offspring = np.empty_like(candidate1)

        # Copy the segment between crossover points from parent1 to offspring
        offspring[a:b] = candidate1[a:b]

        # Map genes from parent1 to parent2 and vice versa
        mapping_p1_to_p2 = dict(zip(candidate1, candidate2))
        mapping_p2_to_p1 = dict(zip(candidate2, candidate1))

        # Fill in the remaining positions using mappings
        for i in range(length):
            if i < a or i >= b:
                gene = candidate2[i]
                while gene in offspring[a:b]:
                    gene = mapping_p1_to_p2[gene]
                offspring[i] = gene

        return offspring

# # Create an instance of the RecombinationOperators class
# operators = RecombinationOperators()

# # Assuming you have 'parents' and 'initial_population' defined earlier
# parents = generate_parent_combinations(initial_population)

# for parent_pair in parents:
#     child_pmx = operators.pmx_crossover(parent_pair[0], parent_pair[1])
#     print('Parent_1:', parent_pair[0])
#     print('Parent_2:', parent_pair[1])
#     print('Child:', child_pmx)
#     check_and_add_child(child_pmx, initial_population)
