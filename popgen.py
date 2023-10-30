import random
import numpy as np

class TSPPopulationGenerator:
    def __init__(self, distance_matrix, population_size):
        """
        Initialize the TSP Population Generator.

        Parameters:
        - distance_matrix: A matrix representing distances between cities.
        - population_size: The desired population size (number of tours).

        This class is used to generate an initial population of tours for the Traveling Salesman Problem (TSP).
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size

    def random_generation(self, random_seed=None):
        """
        Generate an initial population using random permutations of cities.

        Parameters:
        - random_seed: An optional random seed for reproducibility.

        Returns:
        A list of tours, each represented as a random permutation of cities.
        """
        if random_seed is not None:
            random.seed(random_seed)

        initial_population = []
        for _ in range(self.population_size):
            tour = list(range(self.num_cities))
            random.shuffle(tour)  # Create a random permutation
            initial_population.append(tour)
        return initial_population




    def sequential_diversification(self):
        """
        Generate an initial population using sequential diversification.

        Returns:
        A list of tours, each created by starting with a different city and sequentially visiting the rest.
        """
        initial_population = []
        for start_city in range(self.num_cities):
            tour = [start_city] + [city for city in range(self.num_cities) if city != start_city]
            initial_population.append(tour[:])
        return initial_population

    def parallel_diversification(self):
        """
        Generate an initial population using parallel diversification.

        Returns:
        A list of tours, each created by shuffling the order of cities to create diversity.
        """
        initial_population = []
        for _ in range(self.population_size):
            tour = list(range(self.num_cities))
            random.shuffle(tour)
            initial_population.append(tour)
        return initial_population

    def heuristic_initialization(self):
        """
        Generate an initial population using a heuristic approach (nearest neighbor).

        Returns:
        A list of tours, each created by starting with a random city and iteratively selecting the nearest neighbor.
        """
        initial_population = []
        for _ in range(self.population_size):
            unvisited_cities = list(range(self.num_cities))
            current_city = random.choice(unvisited_cities)
            unvisited_cities.remove(current_city)
            tour = [current_city]

            while unvisited_cities:
                nearest_city = min(unvisited_cities, key=lambda city: self.distance_matrix[current_city][city])
                tour.append(nearest_city)
                current_city = nearest_city
                unvisited_cities.remove(current_city)

            initial_population.append(tour)
        return initial_population

# Example usage
file = open('tour50.csv')
distance_matrix = np.loadtxt(file, delimiter=',')
file.close()

population_size = 10
generator = TSPPopulationGenerator(distance_matrix, population_size)

# Generate initial population using random generation
random_population = generator.random_generation(random_seed=42)

# Generate initial population using sequential diversification
sequential_population = generator.sequential_diversification()

# Generate initial population using parallel diversification
parallel_population = generator.parallel_diversification()

# Generate initial population using heuristic initialization
heuristic_population = generator.heuristic_initialization()
