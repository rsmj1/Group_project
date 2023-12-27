import Reporter
import numpy as np
import random
from scipy.stats import truncexpon
import math
import matplotlib.pyplot as plt
import time
import numba as nb



#TODO: 1. Finish two islands - 2. pull out original optimize as secondary optimization function for test - 3. better LSO - 4. Fine tuning, alpha, other hyperparams - 5. Other operators? Than PMX? Than SwapMutation?

# Modify the class name to match your student number.
class TspProg:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
    def optimize(self, filename, params, testFile = None):
		# Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
		
		# Your code here.

        #We can use custom arrays instead of the csv files for testing
        if testFile is not None:
            distanceMatrix = testFile

        #print("Distance Matrix: \n", distanceMatrix)
		#Parameters
        lam = params.la #Population size
        mu = params.mu  #Offspring size  
        k = params.k    #K-tournament selection param  
        iterations = params.iterations
        numCities = len(distanceMatrix[0])
        alpha = 0.2

        bestInd = np.random.permutation(numCities)
        
        ##### GENERATION
        islandIters = 10

        island1Size = int(lam * 0.5)
        island1mu = int(mu * 0.5)
        island2Size = int(lam * 0.5)
        island2mu = int(mu * 0.5)
        island2pressure = 0.99
        exchangeRate = 0.2
        print("Initializing island populations")
        island1pop = nn_krandom_generation(distanceMatrix, island1Size)
        island2pop = nn_krandom_generation(distanceMatrix, island2Size)



        meanHist = []
        minimumHist = []
        i = 0
        yourConvergenceTestsHere = True
        while(yourConvergenceTestsHere):
            it_start = time.time()
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])
            print("Starting Islands")
            island1pop = island1(distanceMatrix, island1pop, islandIters, island1Size, island1mu, k, alpha, numCities)
            island2pop, island2pressure = island2(distanceMatrix, island2pop, islandIters, island2Size, island2mu, island2pressure, alpha, numCities)

            optimizeBestInd(swap_lso, island1pop, distanceMatrix)
            optimizeBestInd(opt2, island2pop, distanceMatrix)


        

            evalPop(island1pop, distanceMatrix, "Island1:")
            evalPop(island2pop, distanceMatrix, "Island2:")

            #Swap individuals based on fitness sharing distances
            
            island1candidateVals = compute_all_shared_fitnesses_island(island1pop, island2pop, distanceMatrix)
            island2candidateVals = compute_all_shared_fitnesses_island(island2pop, island1pop, distanceMatrix)
            i1migrators, i1indices = k_tournament_migration(island1pop, int(island1Size*exchangeRate), island1candidateVals, 5)
            i2migrators, i2indices = k_tournament_migration(island2pop, int(island1Size*exchangeRate), island2candidateVals, 5)

            island1pop[i1indices, :] = i2migrators
            island2pop[i2indices, :] = i1migrators


            population = np.vstack((island1pop, island2pop))

			##### EVALUATION
            objectiveValues = np.array([fitness(ind, distanceMatrix) for ind in population])
            mean = np.mean(objectiveValues)
            minimum = np.min(objectiveValues)
            meanHist.append(mean)
            minimumHist.append(minimum)
            print("Iteration: ", i, ", Mean fitness:", mean, " Min fitness:", minimum, "Mean mutation rate:", alpha)
            i += 1

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
            meanObjective = mean
            bestObjective = minimum
            bestSolution = population[np.argmin(objectiveValues)]
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            it_end = time.time()
            print("Iteration time:", it_end-it_start)
			#print("hello", timeLeft)
            if i >= iterations:
                bestInd = population[np.argmin(objectiveValues)]
                break

            if timeLeft < 0:
                print("No time left, stopping!")
                break
            
        print("Route of best individual:")
        printIndividual(bestInd, alpha, distanceMatrix)
        print("Final mean:", meanHist[len(meanHist)-1], ", Final best:", minimumHist[len(minimumHist)-1])
        plotResuts(meanHist, minimumHist)
		# Your code here.
        return 0
    

    def optimize_old(self, filename, params, testFile = None):
        # Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
		
		# Your code here.

        #We can use custom arrays instead of the csv files for testing
        if testFile is not None:
            distanceMatrix = testFile

        #print("Distance Matrix: \n", distanceMatrix)
		#Parameters
        lam = params.la #Population size
        mu = params.mu  #Offspring size  
        k = params.k    #K-tournament selection param  
        iterations = params.iterations
        numCities = len(distanceMatrix[0])
        alpha = 0.2
        a = 0.9
        selection_pressure = a

        bestInd = np.random.permutation(numCities)
        
        ##### GENERATION
        


        #population = random_generation(distanceMatrix, lam)
        population = nn_krandom_generation(distanceMatrix, lam)
        #routes = parallel_diversification_generation(distanceMatrix, lam)
        #routes = sequential_diversification_generation(distanceMatrix, lam)
        # meanFit = np.mean([fitness(x, distanceMatrix) for x in population])
        # print("start fits:", meanFit)
        # for i in range(lam):
        #     start = time.time()
        #     swap_lso(distanceMatrix, population)
        #     end = time.time()
        #     print("Time:", end - start)
        #     meanFit = np.mean([fitness(x, distanceMatrix) for x in population])
        #     print("new fits:", meanFit)
        
        #LSO after initialization
        fast_swap_lso(distanceMatrix, population)

        meanHist = []
        minimumHist = []
        ##### MUTATION
        mutation = invMutation
        #mutation = swapMutation
        #mutation = insertMutation
        #mutation = scrambleMutation

        i = 0
        yourConvergenceTestsHere = True
        while(yourConvergenceTestsHere):
            it_start = time.time()
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])

			#Create offspring
            offspring = np.empty((mu, numCities), dtype = int)
            num_parents = 2*mu


            ##### SELECTION
            #fitness_values = np.array([fitness(ind, distanceMatrix) for ind in population])
            pop_fitness = np.apply_along_axis(fitness, 1, population, dmatrix=distanceMatrix)

            #selected_individuals = exp_selection(distanceMatrix, population, lam, num_parents) #Version WITHOUT geometric decay
            selected_individuals = exp_selection(distanceMatrix, population, lam, num_parents, pop_fitness, selection_pressure) #Version WITH geometric decay
            #selected_individuals = k_tournament_selection(distanceMatrix, population, num_parents, fitness_values, k)
            #selected_individuals = stochastic_universal_sampling(distanceMatrix, population, num_parents)
            #Geometric decay
            if i % 3 == 0 and a > 0.0001: #Set how aggressive the decay should be
                selection_pressure *= a

            # Select from the population:
            for j in range(mu):
                p1 = selected_individuals[2*j]
                p2 = selected_individuals[2*j + 1]
                ##### RECOMBINATION 
                #offspring[j] = edge_crossover(p1, p2)
                #offspring[j] = partially_mapped_crossover(p1, p2)
                #offspring[j] = pmx(p1, p2, alpha, alpha)
                #offspring[j] = pmx2(p1, p2, alpha, alpha)
                #offspring[j] = pmx2_loop(p1, p2, alpha, alpha)

                offspring[j] = tpx(p1, p2, alpha, alpha)

            mutation(offspring, alpha)
            mutation(population, alpha)

            #LSO
            fast_swap_lso(distanceMatrix, population)

			##### ELIMINATION
            population = elimination(distanceMatrix, population, offspring, lam)
            #population = shared_fitness_elimination(distanceMatrix, population, offspring, lam)

			##### EVALUATION
            objectiveValues = np.array([fitness(ind, distanceMatrix) for ind in population])
            mean = np.mean(objectiveValues)
            minimum = np.min(objectiveValues)
            meanHist.append(mean)
            minimumHist.append(minimum)
            print("Iteration: ", i, ", Mean fitness:", mean, " Min fitness:", minimum, "Mean mutation rate:", alpha)
            i += 1

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
            meanObjective = mean
            bestObjective = minimum
            bestSolution = population[np.argmin(objectiveValues)]
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            it_end = time.time()
            print("Iteration time:", it_end-it_start)
			#print("hello", timeLeft)
            if i >= iterations:
                bestInd = population[np.argmin(objectiveValues)]
                break

            if timeLeft < 0:
                print("No time left, stopping!")
                break
            
        print("Route of best individual:")
        printIndividual(bestInd, alpha, distanceMatrix)
        print("Final mean:", meanHist[len(meanHist)-1], ", Final best:", minimumHist[len(minimumHist)-1])
        plotResuts(meanHist, minimumHist)
		# Your code here.
        return 0

class Parameters:
  def __init__(self, lambd, mu, k, its):
    self.la = lambd
    self.mu = mu
    self.k = k
    self.iterations = its

def evalPop(population, distanceMatrix, name):
    objectiveValues = np.array([fitness(ind, distanceMatrix) for ind in population])
    mean = np.mean(objectiveValues)
    minimum = np.min(objectiveValues)
    print(name, "Mean fitness:", mean, " Min fitness:", minimum)


def optimizeBestInd(optimizer, population, distanceMatrix):
    i = bestIndArg(population, distanceMatrix)
    bestInd = population[i]
    population[i] = optimizer(distanceMatrix, bestInd)

def bestIndArg(population, distanceMatrix):
    fitnesses = np.array([fitness(ind, distanceMatrix) for ind in population])
    return np.argmin(fitnesses)


#Island with:
#Swap Mutation
#PMX
#K-Tournament
#fitness sharing selection
def island1(distanceMatrix, population, iters, lambd, mu, k, alpha, numCities):
    fast_opt2(distanceMatrix, population)
    for i in range(iters):
        it_start = time.time()

        #Create offspring
        offspring = np.empty((mu, numCities), dtype = int)
        num_parents = 2*mu

        ##### SELECTION
        #fitness_values = np.array([fitness(ind, distanceMatrix) for ind in population])
        fitness_values = compute_all_shared_fitnesses(population, distanceMatrix)

        selected_individuals = k_tournament_selection(population, num_parents, fitness_values, k)
        # Select from the population:
        for j in range(mu):
            
            p1 = selected_individuals[2*j]
            p2 = selected_individuals[2*j + 1]
            ##### RECOMBINATION 
            offspring[j] = pmx(p1, p2, alpha, alpha)
        swapMutation(offspring, alpha)
        swapMutation(population, alpha)

        fast_opt2(distanceMatrix, population)
        population = elimination(distanceMatrix, population, offspring, lambd)
        it_end = time.time()
        print(i, "Island1 time:", it_end-it_start)


    return population




#Island with:
#Inverse Mutation
#TPX
#Exp_selection
#fitness sharing selection
def island2(distanceMatrix, population, iters, lambd, mu, selection_pressure, alpha, numCities):
    a = 0.99
    fast_swap_lso(distanceMatrix, population)
    for i in range(iters):
        it_start = time.time()

        #Create offspring
        offspring = np.empty((mu, numCities), dtype = int)
        num_parents = 2*mu

        ##### SELECTION
        #pop_fitness = np.apply_along_axis(fitness, 1, population, dmatrix=distanceMatrix)
        fitness_values = compute_all_shared_fitnesses(population, distanceMatrix)
        #selected_individuals = exp_selection(distanceMatrix, population, lambd, num_parents, fitness_values, selection_pressure) #Version WITH geometric decay
        selected_individuals = k_tournament_selection(population, num_parents, fitness_values, 5)
        if i % 2 == 0 and a > 0.0001: #Set how aggressive the decay should be
            selection_pressure *= a
        else:
            selection_pressure  = 0.2
            
        # Select from the population:
        for j in range(mu):
            p1 = selected_individuals[2*j]
            p2 = selected_individuals[2*j + 1]
            ##### RECOMBINATION 
            offspring[j] = tpx(p1, p2, alpha, alpha)

        invMutation(offspring, alpha)
        invMutation(population, alpha)

        fast_swap_lso(distanceMatrix, population)

        population = elimination(distanceMatrix, population, offspring, lambd)
        it_end = time.time()
        print(i, "Island2 time:", it_end-it_start)


    return population, selection_pressure













def k_tournament_migration(population, num_parents, fitness_values, k=5):
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

    #print("fitness:", fitness_values)
    #fitness_values = compute_all_shared_fitnesses(population, distance_matrix)
    #print("shared:", shared_values)
    selected_parents = []
    selected_indices = []
    for _ in range(num_parents):
        tournament_indices = np.random.choice(len(population), k, replace=False)
        tournament_fitness = fitness_values[tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        while winner_index in selected_indices:
            tournament_indices = np.random.choice(len(population), k, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
        selected_parents.append(population[winner_index])
        selected_indices.append(winner_index)

    return np.array(selected_parents), np.array(selected_indices)






















@nb.njit()
def fitness(individual, dmatrix):
	n = len(dmatrix[0])
	distance = 0
	for i in range(0,n-1):
		distance += dmatrix[individual[i], individual[i+1]]  
	distance += dmatrix[individual[n-1], individual[0]]      
	return distance






################ FITNESS SHARING ##################
'''How to define distance for TSP? 
Look on the internet, how to quantify distance between permutations 
- one could be, how many swaps needed to go from one to the other, 
look at largest common subpath/overlap,
for TSP, it can be difficult to get an actual distance that satisfies triangle inequality. But also not necessarily needed. '''


@nb.njit()
def shared_fitness_elimination(dmatrix, population, offspring, num_survivors):
    individuals = np.vstack((population, offspring))
    n = population.shape[1]
    survivors = np.empty((num_survivors, n), dtype=np.int64)
    num_survivors = survivors.shape[0]

    edges = np.zeros(1000000, dtype=np.int64)
    idx = compute_fitness_vals_best_id(individuals, None, dmatrix, edges)
    survivors[0] = individuals[idx,:]

    for i in range(1, num_survivors):
        edges[:] = 0
        idx = compute_fitness_vals_best_id(individuals, survivors[0:i,:], dmatrix, edges)
        survivors[i] = individuals[idx,:]
    return survivors



@nb.njit()
def compute_fitness_vals_best_id(individuals, survivors, dmatrix, edges):
    num_individuals = individuals.shape[0]
    best_val = np.inf
    best_index = 0
    for j in range(num_individuals):
        fitness_val = shared_fitness(individuals[j], dmatrix, edges, survivors, 1)
        if fitness_val < best_val:
            best_val = fitness_val
            best_index = j
    return int(best_index)


@nb.njit()
def compute_all_shared_fitnesses_island(population1, population2, dmatrix):
    n = population1.shape[0]
    edges = np.zeros(1000000, dtype=np.int64)
    fitnesses = np.empty(n)
    for i in range(n):
        route = population1[i]
        fitnesses[i] = shared_fitness(route, dmatrix, edges, population2)
        edges[:] = 0
    return fitnesses

@nb.njit()
def compute_all_shared_fitnesses(population, dmatrix):
    n = population.shape[0]
    edges = np.zeros(1000000, dtype=np.int64)
    fitnesses = np.empty(n)
    for i in range(n):
        route = population[i]
        fitnesses[i] = shared_fitness(route, dmatrix, edges, population)
        edges[:] = 0
    return fitnesses



@nb.njit()
def shared_fitness(individual, dmatrix, edges, population=None, betaInit=0):
    if population is None:
        return fitness(individual, dmatrix)

    n = individual.shape[0]
    alpha = 1
    sigma =  (n-1) * 0.6 #We need a distance function!

    distances = dist_to_pop(individual, population, edges)
    beta = betaInit
    for i in range(n):
        dist = distances[i]
        if dist <= sigma:
            beta += 1 - (dist/sigma)**alpha
    origFit = fitness(individual, dmatrix)
    res = origFit * beta**np.sign(origFit)
    return res




#I use i+1 as the array is all zeros, and this should not interfere
@nb.njit()
def dist_to_pop(route, pop, edges):
    popSize = pop.shape[0]
    output = np.zeros(popSize, dtype=np.int64)
    for i in range(popSize):
        output[i] = common_edges_dist(route, pop[i], edges, i+1)

    return output


@nb.njit()
def common_edges_dist(route1, route2, edges, num):
    num_edges = 0
    n = route1.shape[0]
    for i in range(n-1):
        a = route1[i]
        b = route1[i+1]
        if a < b:
            edges[a * 1000 + b] = num
        else:
            edges[b * 1000 + a] = num
    if route1[n-1] < route1[0]:
        edges[route1[n-1]*1000+route1[0]] = num
    else:
        edges[route1[0]*1000+route1[n-1]] = num

    for i in range(n-1):
        a = route2[i]
        b = route2[i+1]
        if a < b:            
            val = a * 1000 + b
        else:
            val = b * 1000 + a
        if edges[val] == num:
            num_edges += 1
    a = route2[n-1]
    b = route2[0]
    if a < b:            
        val = a * 1000 + b
    else:
        val = b * 1000 + a
    if edges[val] == num:
        num_edges += 1
        
    # print("DOOOOONNNEEEEEEE EDGES SUM:", sum(edges), len(edges), num, n)
    # for i in range(1000000):
    #     if edges[i] != 0:
    #         print(i, "val:", edges[i])
    return n-num_edges





####################### MUTATIONS #############################

#Mutate inplace, so returns nothing - INVERSION MUTATION (also called (R)reverse (S)equence (M)utation)
@nb.njit()
def invMutation(inds, a):
    for k in range(inds.shape[0]):
        if np.random.uniform() < a:
            i = np.random.randint(0,len(inds[k])-1)
            j = np.random.randint(i+1,len(inds[k]))
            #individual.route[i:j] = individual.route[i:j][::-1]
            inds[k][i:j] = np.flip(inds[k][i:j])


#Mutate inplace, so returns nothing - SWAP MUTATION
@nb.njit()
def swapMutation(inds, a):
  #Mutate with probability prob from the individual
  for k in range(inds.shape[0]):
    if np.random.uniform() < a:
        i = np.random.randint(0,len(inds[k]))
        j = np.random.randint(0,len(inds[k]))
        tmp = inds[k][i]
        inds[k][i] = inds[k][j]
        inds[k][j] = tmp

def scrambleMutation(inds, a):
    """
    Apply Scramble Mutation to an individual in-place.

    Parameters:
    - individual: A NumPy array representing an individual.

    This function mutates the individual in-place, so it returns nothing.
    """
    for k in range(inds.shape[0]):
        if np.random.uniform() < a:
            i = np.random.randint(0, len(inds[k]))
            j = np.random.randint(i + 1, len(inds[k]))
            segment = inds[k][i:j]
            np.random.shuffle(segment)
            inds[k][i:j] = segment

def insertMutation(inds, a):
    """
    Apply Insert Mutation to an individual in-place.

    Parameters:
    - individual: A NumPy array representing an individual.

    This function mutates the individual in-place, so it returns nothing.
    """
    for k in range(inds.shape[0]):
        if np.random.uniform() < a:
            i = np.random.randint(0, len(inds[k]))
            j = np.random.randint(0, len(inds[k]))
            gene = inds[k][i]
            inds[k] = np.insert(inds[k], j, gene)
            if j <= i:
                i += 1
            inds[k] = np.delete(inds[k], i)



######################### RECOMBINATIONS ###############################

#Ordered two-point crossover for permutations
@nb.njit()
def tpx(p1, p2, a1, a2):
    n = len(p1)
    i = np.random.randint(0,n-1)
    j = np.random.randint(i+1,n) 

    route1 = np.empty(n, dtype = np.int64)
    route1[0:i] = p1[0:i]
    route1[j:n] = p1[j:n]
    set1 = set(p1[i:j])

    c = 0
    currSlot = i
    while len(set1) != 0:
        if p2[c] in set1:
            route1[currSlot] = p2[c]
            set1.remove(p2[c])
            currSlot += 1
        c += 1
    return route1



### PMX ###



def pmx(candidate1, candidate2, a1, a2):
    length = candidate1.size
   # Choose our crossover points:
    a = np.random.randint(length-1)
    b = np.random.randint(a+1, length)
    interval = candidate1[a:b]
    offspring = np.empty(length, dtype=np.int64)
    offspring[a:b] = interval
    index_map = np.arange(length)
    index_map[candidate1] = np.arange(length)
    loop_indexes = np.concatenate([np.arange(0,a), np.arange(b, length)])
    offspring = pmxloop(index_map, candidate2, offspring, loop_indexes, interval)

    #print("offspring:", offspring)
    return offspring
@nb.njit()
def pmxloop(index_map, candidate2, offspring, loop_indexes, interval):
    for i in loop_indexes:
        curr_city = candidate2[i]
        while curr_city in interval:
            new_index = index_map[curr_city]
            curr_city = candidate2[new_index]
        offspring[i] = curr_city
    return offspring


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

            initial_population.append(np.array(tour))
        return np.array(initial_population)



########################### Population Generation ######################################
#Inserts (number of cities / choice) * 100 % random cities in random positions (except first random element in position 0), before filling the rest with NN
def nn_krandom_generation(distance_matrix, lam):
    num_cities = len(distance_matrix[0])
    initial_population = np.empty((lam, num_cities), dtype=np.int64)
    num_init = 5


    for new_route in range(lam):
        initial_cities = np.random.choice(num_cities, num_init, False)
        initial_positions = np.random.choice(np.arange(1, num_cities), num_init-1, False)

        #Changed it to a boolean filter to get around having to use .remove/.delete, much faster this way.
        unvisited_cities = np.ones(num_cities, dtype=np.bool_)
        unvisited_cities[initial_cities] = False

        tour = np.full(num_cities, -1)
        tour[0] = initial_cities[0]   
        tour[initial_positions] = initial_cities[1:]
        for i in range(1, num_cities):
            if tour[i] != -1:
                continue
            dists = distance_matrix[tour[i-1], unvisited_cities]
            nearest_city_index = np.argmin(dists)
            nearest_city = np.nonzero(unvisited_cities)[0][nearest_city_index]
            tour[i] = nearest_city
            unvisited_cities[nearest_city] = False
        #if not check_perm(tour):
        #     print("Not working...")
        #     print("The tour:", np.sort(tour))
        #initial_population.append(np.array(tour))
        initial_population[new_route] = tour
    return initial_population






def check_perm(array):
    list_array = array.tolist()
    set_array = set(list_array)
    if len(list_array) != len(set_array):
        return False
    if max(list_array) != len(list_array) - 1:
        return False
    if min(list_array) != 0:
        return False
    return True

#Updated version
def random_generation(distance_matrix, lam):
    num_cities = len(distance_matrix[0])
    inital_population = np.empty((lam, num_cities), dtype=int)
    for i in range(lam):
        tour = np.random.permutation(num_cities)
        inital_population[i,:] = tour
    return inital_population




########################### Selection ################################
def k_tournament_selection(population, num_parents, fitness_values, k=5):
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

    #print("fitness:", fitness_values)
    #fitness_values = compute_all_shared_fitnesses(population, distance_matrix)
    #print("shared:", shared_values)
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


def exp_selection(dmatrix, pop, l, mu, pop_fitness, selection_pressure=0.01):
    # Create the distribution:
    a = math.log(selection_pressure)/(l-1)
    beta = -1/a
    X = truncexpon(b=(l)/beta, loc=0, scale=beta)
    data = X.rvs(mu)
    index = [int(a) for a in np.floor(data)]
    
    order = np.argsort(pop_fitness)
    output = np.array([pop[order[i]] for i in index])

    return output



########################### Elimination ################################

#lambda + mu elimination

def elimination(dmatrix, pop, offspring, l):
    combination = np.vstack((pop, offspring))
    pred = np.array([fitness(x, dmatrix) for x in combination])
    ordering = np.argsort(pred)
    choices = combination[ordering][:l]

    return choices


@nb.njit()
def swap_lso(dmatrix, ind):
    curr_route = ind
    bestFit = fitness(curr_route, dmatrix)
    bestj = -1
    bestk = -1
    for j in range(ind.shape[0]):
        for k in range(j, ind.shape[0]):
            temp = curr_route[j]
            curr_route[j] = curr_route[k]
            curr_route[k] = temp
            fit = fitness(curr_route, dmatrix)
            if fit < bestFit:
                bestFit = fit
                print("BestFit:", bestFit)

                bestj = j
                bestk = k
            #unswap for next iteration
            temp = curr_route[j]
            curr_route[j] = curr_route[k]
            curr_route[k] = temp
    temp = ind[bestj]
    ind[bestj] = ind[bestk]
    ind[bestk] = temp
    return ind

@nb.njit()
def swap3_lso(dmatrix, ind):
    n = ind.shape[0]
    bestival = -1
    bestjval = -1
    bestkval = -1
    besti = -1
    bestj = -1
    bestk = -1
    #i = 0, j = 1, k = 2
    bestFit = fitness(ind, dmatrix)
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                ival = ind[i]
                jval = ind[j]
                kval = ind[k]               
                #Perm1 021
                ind[i] = ival
                ind[j] = kval
                ind[k] = jval
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestival = ival
                    bestjval = kval
                    bestkval = jval
                    besti, bestj, bestk = i,j,k
                #Perm2 102
                ind[i] = jval
                ind[j] = ival
                ind[k] = kval
                if fit < bestFit:
                    bestFit = fit
                    bestival = jval
                    bestjval = ival
                    bestkval = kval
                    besti, bestj, bestk = i,j,k
                #Perm3 120
                ind[i] = jval
                ind[j] = kval
                ind[k] = ival    
                if fit < bestFit:
                    bestFit = fit
                    bestival = jval
                    bestjval = kval
                    bestkval = ival
                    besti, bestj, bestk = i,j,k
                #Perm4 201
                ind[i] = kval
                ind[j] = ival
                ind[k] = jval    
                if fit < bestFit:
                    bestFit = fit
                    bestival = kval
                    bestjval = ival
                    bestkval = jval
                    besti, bestj, bestk = i,j,k   
                #Perm5 210
                ind[i] = kval
                ind[j] = jval
                ind[k] = ival    
                if fit < bestFit:
                    bestFit = fit
                    bestival = kval
                    bestjval = jval
                    bestkval = ival
                    besti, bestj, bestk = i,j,k
                #Reset individual
                ind[i] = ival
                ind[j] = jval
                ind[k] = kval
    #Perform best 3-swap
    ind[besti] = bestival
    ind[bestj] = bestjval
    ind[bestk] = bestkval
    print("best found fit:", bestFit)
    print("fit of result individual:", fitness(ind, dmatrix))


@nb.njit()
def fast_swap_lso(dmatrix, pop):
    n = pop.shape[1]
    for i in range(pop.shape[0]):
        curr_route = pop[i]
        bestFit = fitness(curr_route, dmatrix)
        bestj = -1
        bestk = -1
        iters = 5000
        for j in range(iters):
            idi = np.random.randint(n-1)
            idj = np.random.randint(idi+1, n)
            temp = curr_route[idi]
            curr_route[idi] = curr_route[idj]
            curr_route[idj] = temp
            fit = fitness(curr_route, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestj = idi
                bestk = idj
            #unswap for next iteration
            temp = curr_route[idi]
            curr_route[idi] = curr_route[idj]
            curr_route[idj] = temp
        temp = pop[i, bestj]
        pop[i,bestj] = pop[i,bestk]
        pop[i,bestk] = temp 

@nb.njit()
def opt2(dmatrix, ind):
    curr_route = ind
    bestFit = fitness(curr_route, dmatrix)
    bestj = -1
    bestk = -1
    for j in range(ind.shape[0]):
        for k in range(j, ind.shape[0]):
            ind[j:k] = np.flip(ind[j:k])
            fit = fitness(curr_route, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestj = j
                bestk = k
            #unswap for next iteration
            ind[j:k] = np.flip(ind[j:k])

    ind[bestj:bestk] = np.flip(ind[bestj:bestk])
    print("BestFit:", bestFit)
    return ind


@nb.njit()
def fast_opt2(dmatrix, pop):
    for k in range(pop.shape[0]):
        curr_route = pop[k]
        bestFit = fitness(curr_route, dmatrix)
        bestj = -1
        bestk = -1
        iters = 5000
        for a in range(iters):
            i = np.random.randint(0,len(pop[k])-1)
            j = np.random.randint(i+1,len(pop[k]))
            #individual.route[i:j] = individual.route[i:j][::-1]
            pop[k][i:j] = np.flip(pop[k][i:j])
            fit = fitness(curr_route, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestj = i
                bestk = j
            #Undo operation
            pop[k][i:j] = np.flip(pop[k][i:j])
        pop[k][bestj:bestk] = np.flip(pop[k][bestj:bestk])


def opt3(dmatrix, ind):
    n = ind.shape[0]
    besti = -1
    bestj = -1
    bestk = -1
    #i = 0, j = 1, k = 2
    bestFit = fitness(ind, dmatrix)
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                ival = ind[i]
                jval = ind[j]
                kval = ind[k]               
                #Perm1 i,k
                ind[i:k] = np.flip(ind[i:k])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    besti, bestj, bestk = i,j,k
                ind[i:k] = np.flip(ind[i:k])

                #Perm2 j,k
                ind[j:k] = np.flip(ind[j:k])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    besti, bestj, bestk = i,j,k
                ind[j:k] = np.flip(ind[j:k])

                #Perm3 i,j
                ind[i:j] = np.flip(ind[i:j])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    besti, bestj, bestk = i,j,k
                ind[i:j] = np.flip(ind[i:j])

                #Perm4 ij, jk
                ind[i:j] = np.flip(ind[i:j])
                ind[j:k] = np.flip(ind[j:k])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    besti, bestj, bestk = i,j,k
                ind[j:k] = np.flip(ind[j:k])
                ind[i:j] = np.flip(ind[i:j])

                #Perm5 ik, jk
                ind[i:k] = np.flip(ind[i:k])
                ind[j:k] = np.flip(ind[j:k])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    besti, bestj, bestk = i,j,k
                ind[j:k] = np.flip(ind[j:k])
                ind[i:k] = np.flip(ind[i:k])
                #Perm6 ik, ij
                ind[i:k] = np.flip(ind[i:k])
                ind[i:j] = np.flip(ind[i:j])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    besti, bestj, bestk = i,j,k
                ind[i:j] = np.flip(ind[i:j])
                ind[i:k] = np.flip(ind[i:k])
                #Perm7 ik, ij, jk
                ind[i:k] = np.flip(ind[i:k])
                ind[i:j] = np.flip(ind[i:j])
                ind[j:k] = np.flip(ind[j:k])
                
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    besti, bestj, bestk = i,j,k
                ind[j:k] = np.flip(ind[j:k])
                ind[i:j] = np.flip(ind[i:j])
                ind[i:k] = np.flip(ind[i:k])

    #Perform best 3-swap
    ind[besti] = bestival
    ind[bestj] = bestjval
    ind[bestk] = bestkval
    print("best found fit:", bestFit)
    print("fit of result individual:", fitness(ind, dmatrix))
    

#TODO: use fitness sharing for selection for island2
#TODO: Implement 3-opt exhaustive for one individual


######################## Mutation Rate Related Stuff #########################
def combineAlphas(a1, a2):
    b = 2 * np.random.uniform() - 0.5 #Between -0.5 and 1.5
    a = a1 + b * (a2-a1) 
    return np.abs(a)


######################## MISC HELPER FUNCTIONS ############################

#At some point we need to translate our path into a cycle
def solutionToCycle(path):
    return []

def printIndividual(route, alpha, dmatrix = None):
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



# testArray2 = np.array([[0, 1.5, 2.4, 3.4],
#                        [2.8, 0, np.inf, 1.3],
# 					   [10., 5.4, 0, 9.5],
#                        [np.inf, 3.8, 9.3, 0]])


# testL = 5000
# testM = 2500
# file = open("tour50.csv")
# distanceMatrix = np.loadtxt(file, delimiter=",")
# #distanceMatrix[distanceMatrix==np.inf] = 0
# file.close()
# nn_krandom_generation(distanceMatrix, 1000)
# testPop = random_generation(distanceMatrix, testL)
# testOff = random_generation(distanceMatrix, testL)
# testPop2 = np.empty(testL, dtype = Individual)
# testOff2 = np.empty(testL, dtype = Individual)
# for ro, route in enumerate(testPop):
#         testPop2[ro] = Individual(route=route)
# for ro1, route1 in enumerate(testOff):
#         testOff2[ro1] = Individual(route=route1)

#testArray = np.array([1,2,3,4,5,6,7,8])
#testIndex = [2,2,3,4,5]
#print("Indexing:", testArray[testIndex])

#tpx(testInd, testInd2)
# a = np.array([0,1,2,3,4,5,6,7,8])
# b = np.array([8,2,6,7,1,5,4,0,3])
# c = np.arange(9)
# d = np.zeros(9, dtype=np.int64) +9
# d[3:7] = a[3:7]
# print("a  :", a)
# print("b  :", b)
# print("slic", d)
# print("idx:", c)
# print("pmx:", pmx(a, b, 0, 0)+1)
# print("pmx2", pmx2(a,b,0,0)+1)
# print("ploo", pmx2_loop(a,b,0,0)+1)
    
# a = np.array([0,1,2,3,4,5,6,7,8])
# b = np.array([1,2,6,7,8,5,4,3,0])
# c = np.array([[1,2,6,7,8,5,4,3,0], [0,1,2,3,4,5,6,7,8], [8,7,6,5,4,3,2,1,0]])
# edges = np.empty(1000000)
# print("dist:", common_edges_dist(a, b, edges, 1))

testArray1 = np.array([[0, 1.5, 2.4, 3.4],
                       [2.8, 0, 5.1, 1.3],
					   [10., 5.4, 0, 9.5],
                       [6.6, 3.8, 9.3, 0]])
    
population = np.array([[0,1,2,3],[3,1,2,0]])
a = np.array([0,1,2,3])
optimizeBestInd(swap_lso, population, testArray1)
# print("pop:", population)
# swap_lso(testArray1, population)
# print("pop:", population)

#print("dists", dist_to_pop(a,c))

prog = TspProg()
params = Parameters(lambd=500, mu=500, k=5, its=1000)
prog.optimize("tour50.csv", params)
#prog.optimize_old("tour200.csv", params)





# tour50: simple greedy heuristic 27723
# 7.5%: 25643
# tour100: simple greedy heuristic 90851
# 7.5%: 84037
# tour200: simple greedy heuristic 39745
# 7.5%: 36764
# tour500: simple greedy heuristic 157034
# 7.5%: 145256
# tour750: simple greedy heuristic 197541
# 7.5%: 182725
# tour1000: simple greedy heuristic 195848
# 7.5%: 181159
