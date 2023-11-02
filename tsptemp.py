import Reporter
import numpy as np
import random
from scipy.stats import truncexpon
import math
import matplotlib.pyplot as plt

# Modify the class name to match your student number.
class TspProg:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
    def optimize(self, filename, params, oneOffspring = True, testFile = None):
		# Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
		
		# Your code here.

        #We can use custom arrays instead of the csv files for testing
        if testFile is not None:
            distanceMatrix = testFile

        print("Distance Matrix: \n", distanceMatrix)
		#Parameters
        lam = params.la #Population size
        mu = params.mu  #Offspring size  
        k = params.k    #K-tournament selection param  
        iterations = params.iterations
        numCities = len(distanceMatrix[0])

        bestInd = Individual(numCities)
        
        ##### GENERATION
        population = np.empty(mu, dtype = Individual)
        #routes = heuristic_generation(distanceMatrix, lam)
        routes = random_generation(distanceMatrix, lam)
        #routes = parallel_diversification_generation(distanceMatrix, lam)
        #routes = sequential_diversification_generation(distanceMatrix, lam)
        for ro, route in enumerate(routes):
            population[ro] = Individual(route=route)

        meanHist = []
        minimumHist = []

        ##### MUTATION
        mutation = invMutation
        #mutation = swapMutation
        #mutation = insertMutation
        #mutation = scrambleMutation

        i = 0
        yourConvergenceTestsHere = True
        while( yourConvergenceTestsHere ):
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])

			# Your code here.

			#Create offspring
            offspring = np.empty(mu, dtype = Individual)

            num_parents = mu
            if oneOffspring:
                num_parents = 2*mu

            selected_individuals = exp_selection(distanceMatrix, population, lam, num_parents)
            #selected_individuals = k_tournament_selection(distanceMatrix, population, num_parents)
            #selected_individuals = stochastic_universal_sampling(distanceMatrix, population, num_parents)

            # Select from the population:
            if oneOffspring: #Recombination resulting in one offspring
                for j in range(mu):
                    p1 = selected_individuals[2*j]
                    p2 = selected_individuals[2*j + 1]
                    ##### RECOMBINATION 
                    #offspring[j] = edge_crossover(p1, p2)
                    #offspring[j] = recursive_fill(p1, p2)
                    #offspring[j] = partially_mapped_crossover(p1, p2)
                    #offspring[j] = order_crossover(p1, p2) #TODO: Fix
                    #offspring[j] = cycle_crossover(p1, p2) #TODO: Fix
                    offspring[j] = pmx(p1, p2)
                    mutation(offspring[j]) #Invmutation generally has worse performance than swap
            else: #Recombination resulting in two offspring
                for j in range(mu//2):
                    p1 = selected_individuals[2*j]
                    p2 = selected_individuals[2*j + 1]
                    ##### RECOMBINATION
                    res = tpx(p1, p2)
                    offspring[j*2] = res[0]
                    offspring[j*2+1] = res[1]
                    mutation(offspring[j*2]) #Invmutation generally has worse performance than swap
                    mutation(offspring[j*2+1]) #Invmutation generally has worse performance than swap

            for elem in population:
                mutation(elem)

			##### ELIMINATION
            population = elimination(distanceMatrix, population, offspring, lam)

			##### EVALUATION
            objectiveValues = np.array([fitness(distanceMatrix, ind) for ind in population])
            mean = np.mean(objectiveValues)
            minimum = np.min(objectiveValues)
            meanHist.append(mean)
            minimumHist.append(minimum)
            print("Iteration: ", i, ", Mean fitness:", mean, " Min fitness:", minimum, "Mean mutation rate:", np.mean(np.array([ind.alpha for ind in population])))
            i += 1

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
            meanObjective = mean
            bestObjective = minimum
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			#print("hello", timeLeft)
            if i >= iterations:
                bestInd = population[np.argmin(objectiveValues)]
                break

            if timeLeft < 0:
                break
            
        print("Route of best individual:")
        printIndividual(bestInd, distanceMatrix)
        print("Final mean:", meanHist[len(meanHist)-1], ", Final best:", minimumHist[len(minimumHist)-1])
        plotResuts(meanHist, minimumHist)
		# Your code here.
        return 0


class Individual:
  #Individuals from the population for the algorithm
  #Each will be initialized randomly and represent a permutation of cities in the TSP problem
  def __init__(self, numCities = None, route = None, alpha = None):
    if route is None:
      self.route = np.random.permutation(numCities)
    else:
      self.route = route
    #self adaptivity parameter
    if alpha is None:
      #self.alpha = max(0.05, 0.1 + 0.05*np.random.normal()) # max(0.05, 0.1 + ~N(0, 0.05^2))
      self.alpha = 0.2
    else:
      self.alpha = alpha

class Parameters:
  def __init__(self, lambd, mu, k, its):
    self.la = lambd
    self.mu = mu
    self.k = k
    self.iterations = its


def fitness(dmatrix, individual):
	distance = 0
	n = len(dmatrix[0])
	route = individual.route
	#print("Route length:", len(route))
	for i in range(0,n-1):
		distance += dmatrix[route[i], route[i+1]]  
	distance += dmatrix[route[n-1], route[0]]      
	return distance


####################### MUTATIONS #############################

#Mutate inplace, so returns nothing - INVERSION MUTATION (also called (R)reverse (S)equence (M)utation)
def invMutation(individual):
  #Mutate with probability prob from the individual
  if np.random.uniform() < individual.alpha:
    i = np.random.randint(0,len(individual.route)-1)
    j = np.random.randint(i+1,len(individual.route))
    #individual.route[i:j] = individual.route[i:j][::-1]
    individual.route[i:j] = np.flip(individual.route[i:j])



#Mutate inplace, so returns nothing - SWAP MUTATION
def swapMutation(individual):
  #Mutate with probability prob from the individual
  if np.random.uniform() < individual.alpha:
    i = np.random.randint(0,len(individual.route))
    j = np.random.randint(0,len(individual.route))
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



######################### RECOMBINATIONS ###############################

#Placeholder function that does nothing
def tpx(parent1, parent2):
    i = np.random.randint(0,len(parent1.route)-1)
    j = np.random.randint(i+1,len(parent1.route)) 
    route1 = np.ndarray.copy(parent1.route)
    route2 = np.ndarray.copy(parent2.route)
    temp = np.ndarray.copy(route1[i:j])
    route1[i:j] = np.ndarray.copy(route2[i:j])
    route2[i:j] = temp
    alpha = combineAlphas(parent1.alpha, parent2.alpha)
    return Individual(route=route1, alpha=alpha), Individual(route=route2, alpha=alpha)

def partially_mapped_crossover(parent1, parent2):
    """
    Perform Partially Mapped Crossover (PMX) on two individuals.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        Individual: A new individual created through PMX.
    """
    candidate1 = list(parent1.route)
    candidate2 = list(parent2.route)
    if len(candidate1) != len(candidate2):
        print('Candidate solutions must have the same length')
        return 0

    length = len(candidate1)
    index_set = set()
    full_set = set(range(length))

    # Choose our crossover points:
    my_range = np.random.choice(length, 2, False)
    a = min(my_range)
    b = max(my_range)

    for j in range(a, b):
        index_set.add(j)

    # Initialize an empty offspring:
    offspring = [None] * (length - 1)

    if a > b:
        middle_section1 = candidate1[b:a + 1]
        middle_section2 = candidate2[b:a + 1]

        offspring[b:a] = middle_section1
        for j in range(b, a + 1):
            index_set.add(j)
    elif b > a:
        middle_section1 = candidate1[a:b + 1]
        middle_section2 = candidate2[a:b + 1]

        offspring[a:b] = middle_section1
        for j in range(a, b):
            index_set.add(j + 1)

    for count, item in enumerate(middle_section2):
        if item not in set(middle_section1):
            item2 = middle_section1[count]
            index = candidate2.index(item2)
            recursive_fill(index, index_set, item, offspring, candidate2)

    for final_item in full_set ^ index_set:
        offspring[final_item] = candidate2[final_item]

    return Individual(route=np.array(offspring), alpha=combineAlphas(parent1.alpha, parent2.alpha))

def recursive_fill(index, index_set, item, offspring, candidate):
    if index not in index_set:
        offspring[index] = item
        index_set.add(index)
    else:
        new_item = offspring[index]
        new_index = candidate.index(new_item)
        recursive_fill(new_index, index_set, item, offspring, candidate)


def edge_crossover(parent1, parent2):
    """
    Perform Edge Crossover on two individuals.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        Individual: A new individual created through Edge Crossover.
    """
    candidate1 = list(parent1.route)
    candidate2 = list(parent2.route)
    if len(candidate1) != len(candidate2):
        print('Candidate solutions must have the same length')
        return 0

    length = len(candidate1)
    neighbor_lists = {}

    for node in range(length):
        neighbor_lists[node] = set([candidate1[(node + 1) % length], candidate1[(node - 1) % length]])

    child = [None] * length
    current_node = candidate1[0]

    for i in range(length):
        child[i] = current_node
        for neighbor in neighbor_lists:
            neighbor_lists[neighbor].discard(current_node)
        next_nodes = list(neighbor_lists[current_node])

        if next_nodes:
            current_node = np.random.choice(next_nodes)
        else:
            remaining_nodes = set(candidate1) - set(child)
            if remaining_nodes:
                current_node = np.random.choice(list(remaining_nodes))

    return Individual(route=np.array(child), alpha=combineAlphas(parent1.alpha, parent2.alpha))


#TODO: Fix
def order_crossover(self, crossover_rate=0.9):
        """
        Apply Order Crossover (OX1) to pairs of chromosomes.

        Parameters:
        - crossover_rate: The probability of applying crossover to a pair.

        Returns:
        A new population resulting from OX1.
        """
        new_population = []

        def order_crossover_single(parent1, parent2):
            if random.random() > crossover_rate:
                return parent1, parent2

            length = len(parent1)
            cut1, cut2 = sorted(random.sample(range(length), 2))

            child1 = [-1] * length
            child2 = [-1] * length

            # Copy the segment between the cuts directly
            child1[cut1:cut2] = parent1[cut1:cut2]
            child2[cut1:cut2] = parent2[cut1:cut2]

            # Map the rest of the genes
            pointer1, pointer2 = cut2, cut2
            for i in range(cut2, cut2 + length):
                i %= length
                if parent2[i] not in child1:
                    child1[pointer1] = parent2[i]
                    pointer1 += 1
                if parent1[i] not in child2:
                    child2[pointer2] = parent1[i]
                    pointer2 += 1

            return child1, child2

        for i in range(0, len(self.population), 2):
            child1, child2 = order_crossover_single(self.population[i], self.population[i + 1])
            new_population.extend([child1, child2])

        return new_population


#TODO: Fix
def cycle_crossover(self, crossover_rate=0.9):
        """
        Apply Cycle Crossover to pairs of chromosomes.

        Parameters:
        - crossover_rate: The probability of applying crossover to a pair.

        Returns:
        A new population resulting from Cycle Crossover.
        """
        new_population = []

        def cycle_crossover_single(parent1, parent2):
            if random.random() > crossover_rate:
                return parent1, parent2

            length = len(parent1)
            child1 = [-1] * length
            child2 = [-1] * length

            # Initialize a list to keep track of visited indices
            visited = [False] * length
            cycles = []

            for i in range(length):
                if not visited[i]:
                    cycle = []
                    j = i
                    while True:
                        cycle.append(j)
                        visited[j] = True
                        j = parent2.index(parent1[j])
                        if j == i:
                            break
                    cycles.append(cycle)

            for i, cycle in enumerate(cycles):
                if i % 2 == 0:
                    for j in cycle:
                        child1[j] = parent1[j]
                        child2[j] = parent2[j]
                else:
                    for j in cycle:
                        child1[j] = parent2[j]
                        child2[j] = parent1[j]

            return child1, child2

        for i in range(0, len(self.population), 2):
            child1, child2 = cycle_crossover_single(self.population[i], self.population[i + 1])
            new_population.extend([child1, child2])

        return new_population


### PMX ###

def pmx(candidate11, candidate22):
    candidate1 = list(candidate11.route)
    candidate2 = list(candidate22.route)
    if len(candidate1) != len(candidate2):
        print('Candidate solutions must have same length')
        return 0

    length = len(candidate1)
    index_set = set()
    full_set = set(range(length))

    # Choose our crossover points:

    my_range = np.random.choice(length,(2),False)
    a = my_range[0]
    b = my_range[1]

    #print(a,b)
    for j in range(a,b):
        index_set.add(j)

    # Initialise an empty offspring:

    offspring = [None] * (length-1)

    # Now lets fill up the offspring:

    if a>b:
        middle_section1 = candidate1[b:a+1]
        middle_section2 = candidate2[b:a+1]

        offspring[b:a] = middle_section1
        for j in range(b,a+1):
         index_set.add(j)

    elif b>a:
        middle_section1 = candidate1[a:b+1]
        middle_section2 = candidate2[a:b+1]

        offspring[a:b] = middle_section1
        for j in range(a,b):
            index_set.add(j+1)

    for count, item in enumerate(middle_section2):
        if item not in set(middle_section1):
            # Find which item is in its place:
            item2 = middle_section1[count]
            # Now find where in candidate2 does item2 reside:
            index = candidate2.index(item2)
            # We replace offspring[index] with the original item IF IT ISN'T ALREADY OCCUPIED:
            recursive_fill(index,index_set,item,offspring,candidate2)

    # Finally, fill in the blanks:
    for final_item in full_set^index_set:
        offspring[final_item] = candidate2[final_item]

    return Individual(route = np.array(offspring), alpha=combineAlphas(candidate11.alpha, candidate22.alpha))

### END PMX

########################### Population Generation ######################################
def heuristic_generation(distance_matrix, lam):
        """
        Generate an initial population using a heuristic approach (nearest neighbor).

        Returns:
        A list of tours, each created by starting with a random city and iteratively selecting the nearest neighbor.
        """
        num_cities = len(distance_matrix[0])
        initial_population = []
        for _ in range(lam):
            unvisited_cities = list(range(num_cities))
            current_city = random.choice(unvisited_cities)
            unvisited_cities.remove(current_city)
            tour = [current_city]

            while unvisited_cities:
                nearest_city = min(unvisited_cities, key=lambda city: distance_matrix[current_city][city])
                tour.append(nearest_city)
                current_city = nearest_city
                unvisited_cities.remove(current_city)

            initial_population.append(tour)
        return initial_population

def random_generation(distance_matrix, lam, random_seed=None):
        """
        Generate an initial population using random permutations of cities.

        Parameters:
        - random_seed: An optional random seed for reproducibility.

        Returns:
        A list of tours, each represented as a random permutation of cities.
        """
        if random_seed is not None:
            random.seed(random_seed)
        num_cities = len(distance_matrix[0])

        initial_population = []
        for _ in range(lam):
            tour = list(range(num_cities))
            random.shuffle(tour)  # Create a random permutation
            initial_population.append(tour)
        return initial_population


def sequential_diversification_generation(distance_matrix, lam):
    """
    Generate an initial population using sequential diversification.

    Returns:
    A list of tours, each created by starting with a different city and sequentially visiting the rest.
    """
    initial_population = []
    num_cities = len(distance_matrix[0])

    for start_city in range(num_cities):
        tour = [start_city] + [city for city in range(num_cities) if city != start_city]
        initial_population.append(tour[:])
    return initial_population

def parallel_diversification_generation(distance_matrix, lam):
    """
    Generate an initial population using parallel diversification.

    Returns:
    A list of tours, each created by shuffling the order of cities to create diversity.
    """
    initial_population = []
    num_cities = len(distance_matrix[0])

    for _ in range(lam):
        tour = list(range(num_cities))
        random.shuffle(tour)
        initial_population.append(tour)
    return initial_population



########################### Selection ################################

def k_tournament_selection(distance_matrix, population, num_parents, k=5):
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
    if not isinstance(population, np.ndarray):
        raise ValueError("Inputs 'population' must be numpy arrays.")
    if num_parents <= 0 or num_parents > 2*len(population) or k <= 0:
        raise ValueError("Invalid number of parents or tournament size.")

    fitness_values = np.array([fitness(distance_matrix, ind) for ind in population])

    selected_parents = []
    for _ in range(num_parents):
        tournament_indices = np.random.choice(len(population), k, replace=False)
        tournament_fitness = fitness_values[tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        selected_parents.append(population[winner_index])

    return np.array(selected_parents)


def stochastic_universal_sampling(distance_matrix, population, num_parents):
    """
    Perform Stochastic Universal Sampling (SUS) to select parents from the population.

    Args:
        distance_matrix (numpy.ndarray): The matrix of distances between cities in the TSP instance
        population (numpy.ndarray): The population to select from.
        num_parents (int): The number of parents to select.

    Returns:
        numpy.ndarray: An array of selected parents from the population.
    """
    if not isinstance(population, np.ndarray):
        raise ValueError("Inputs 'population' must be a numpy array.")
    if num_parents <= 0 or num_parents > 2*len(population):
        raise ValueError("Invalid number of parents to select.")

    fitness_values = np.array([fitness(distance_matrix, ind) for ind in population])
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

def exp_selection(dmatrix, pop, l, mu, selection_pressure=0.0001):
    # Create the distribution:
    a = math.log(selection_pressure)/(l-1)
    beta = -1/a
    X = truncexpon(b=(l)/beta, loc=0, scale=beta)
    data = X.rvs(mu)

    pop_dict = {}

    for ind in pop:
        pop_dict[ind] = fitness(dmatrix,ind)

    sorted_pop_list = sorted(pop_dict.items(), key=lambda x:x[1])

    index = [int(a) for a in np.floor(data)]

    output = [sorted_pop_list[i][0] for i in index]

    return output



########################### Elimination ################################

#lambda + mu elimination
def elimination(dmatrix, pop, offspring, l):
    combination = np.append(pop, offspring)
	#print('Combo: ',combination)
    pred = np.array([fitness(dmatrix, x) for x in combination])
	#print("Pred:", pred)
    ordering = np.argsort(pred)
	#print("Ordering:", ordering)
    choices = combination[ordering][:l]
	#print("Choices:", choices)
    return choices


######################## Mutation Rate Related Stuff #########################
def combineAlphas(a1, a2):
    b = 2 * np.random.uniform() - 0.5 #Between -0.5 and 1.5
    a = a1 + b * (a2-a1) 
    return np.abs(a)


######################## MISC HELPER FUNCTIONS ############################

#At some point we need to translate our path into a cycle
def solutionToCycle(path):
    return []

def printIndividual(ind, dmatrix = None):
    route = ind.route
    alpha = ind.alpha
	#print("Alpha: ", alpha)
    print("Route: ", end="")
    if dmatrix is None:
        for i, r in enumerate(route):
            print(r, end = " -> ")
    else: 
        for i in range(len(route)-1):
            print(route[i], "->", route[i+1], "(", dmatrix[route[i], route[i+1]],")", end="|")
        print(route[len(route)-1], "->", route[0], "(", dmatrix[route[len(route)-1], route[0]],")")

def printPopulation(pop, dmatrix):
     print("Printing population of size ", len(pop))
     for i, ind in enumerate(pop):
          print(i, end= ":")
          printIndividual(ind, dmatrix)
          print("Fitness:", fitness(dmatrix, ind))

def plotResuts(mean, min):
    n = len(mean)
    x = np.arange(n)
    plt.plot(x, mean, 'b', label="Mean Fitness")
    plt.plot(x, min, 'r', label="Min Fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend(loc="upper right")
    plt.show()


############################## RUN SETUP ################################

testArray1 = np.array([[0, 1.5, 2.4, 3.4],
                       [2.8, 0, 5.1, 1.3],
					   [10., 5.4, 0, 9.5],
                       [6.6, 3.8, 9.3, 0]])

testArray2 = np.array([[0, 1.5, 2.4, 3.4],
                       [2.8, 0, np.inf, 1.3],
					   [10., 5.4, 0, 9.5],
                       [np.inf, 3.8, 9.3, 0]])
testInd = Individual(4)
prog = TspProg()
params = Parameters(lambd=500, mu=500, k=5, its=25)
prog.optimize("tour50.csv", params, oneOffspring=True)
