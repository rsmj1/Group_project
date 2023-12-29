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
        islandIters = 5

        island1Size = int(lam/3)
        island1mu = int(mu/3)
        island2Size = int(lam/3)
        island2mu = int(mu/3)
        island2pressure = 0.99
        exchangeRate = 0.05
        print("Initializing island populations")
        #island1pop = nn_krandom_generation(distanceMatrix, island1Size)
        island1pop = random_less_inf_gen(distanceMatrix, island1Size)
        island2pop = nn_krandom_generation(distanceMatrix, island2Size)
        island3pop = nn_krandom_generation(distanceMatrix, island2Size)


        dmatrixmod = distanceMatrix.copy()
        np.fill_diagonal(dmatrixmod, np.inf)
        nns = np.argmin(dmatrixmod, axis=1)

        meanHist = []
        minimumHist = []
        i = 0
        yourConvergenceTestsHere = True
        while(yourConvergenceTestsHere):
            it_start = time.time()
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])


            island1pop = island1(distanceMatrix, island1pop, islandIters, island1Size, island1mu, k, alpha, numCities, nns)
            island2pop, island2pressure = island2(distanceMatrix, island2pop, islandIters, island2Size, island2mu, island2pressure, alpha, numCities)
            island3pop = island3(distanceMatrix, island3pop, islandIters, island1Size, island1mu, k, alpha, numCities, nns)

            evalPop(island1pop, distanceMatrix, "Island1:")
            evalPop(island2pop, distanceMatrix, "Island2:")
            evalPop(island3pop, distanceMatrix, "Island3:")

            #Ring topology
            island1candidateVals = compute_all_shared_fitnesses_island(island1pop, island2pop, distanceMatrix)
            island2candidateVals = compute_all_shared_fitnesses_island(island2pop, island3pop, distanceMatrix)
            island3candidateVals = compute_all_shared_fitnesses_island(island3pop, island1pop, distanceMatrix)

            kmig = 5
            i1migrators, i1indices = k_tournament_migration(island1pop, int(island1Size*exchangeRate), island1candidateVals, kmig)
            i2migrators, i2indices = k_tournament_migration(island2pop, int(island1Size*exchangeRate), island2candidateVals, kmig)
            i3migrators, i3indices = k_tournament_migration(island3pop, int(island1Size*exchangeRate), island3candidateVals, kmig)
            island1pop[i1indices, :] = i3migrators
            island2pop[i2indices, :] = i1migrators
            island3pop[i3indices, :] = i2migrators
            bestInd1 = optimizeBestInd(fast_opt3_lso, island1pop, distanceMatrix)
            bestInd2 = optimizeBestInd(swap_lso, island2pop, distanceMatrix)
            bestInd3 = optimizeBestInd(fast_swap3_lso, island3pop, distanceMatrix)

            i1s = worstIndsArgs(island1pop, distanceMatrix, 1)
            i2s = worstIndsArgs(island2pop, distanceMatrix, 1)
            i3s = worstIndsArgs(island3pop, distanceMatrix, 1)

            island1pop[i1s,:] = bestInd3
            island2pop[i2s,:] = bestInd1
            island3pop[i3s,:] = bestInd2

            population = np.vstack((island1pop, island2pop, island3pop))

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
            if i >= iterations:
                bestInd = population[np.argmin(objectiveValues)]
                break

            if timeLeft < 0:
                print("No time left, stopping!")
                break
            

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
    return optimizer(distanceMatrix, bestInd)

def bestIndArg(population, distanceMatrix):
    fitnesses = np.array([fitness(ind, distanceMatrix) for ind in population])
    return np.argmin(fitnesses)

def worstIndArg(population, distanceMatrix):
    fitnesses = np.array([fitness(ind, distanceMatrix) for ind in population])
    return np.argmax(fitnesses)

def worstIndsArgs(population, distanceMatrix, n):
    fitnesses = np.array([fitness(ind, distanceMatrix) for ind in population])
    return np.argpartition(fitnesses, -n)[-n:]


#Island with:
#Swap Mutation
#PMX
#K-Tournament
#fitness sharing selection
def island1(distanceMatrix, population, iters, lambd, mu, k, alpha, numCities, nns):
    alpha = 0.5
    fast_swap_lso(distanceMatrix, population)
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
        IRGIBNNM(offspring, alpha, nns)        
        RGIBNNM(population, alpha, nns)
        for i in range(5):
            fast_swap3_lso_pop(distanceMatrix, population)
        fast_swap_lso(distanceMatrix, offspring)

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
    alpha = 0.2
    a = 0.99
    fast_swap_lso(distanceMatrix, population)
    for i in range(iters):
        it_start = time.time()

        #Create offspring
        offspring = np.empty((mu, numCities), dtype = int)
        num_parents = 2*mu

        ##### SELECTION
        #fitness_values = np.apply_along_axis(fitness, 1, population, dmatrix=distanceMatrix)
        fitness_values = compute_all_shared_fitnesses(population, distanceMatrix)
        selected_individuals = exp_selection(distanceMatrix, population, lambd, num_parents, fitness_values, selection_pressure) #Version WITH geometric decay
        #selected_individuals = k_tournament_selection(population, num_parents, fitness_values, 5)
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

        fast_opt2_lso(distanceMatrix, population)
        fast_opt2_lso(distanceMatrix, offspring)

        population = elimination(distanceMatrix, population, offspring, lambd)
        it_end = time.time()
        print(i, "Island2 time:", it_end-it_start)

    return population, selection_pressure



#Island with:
#Inverse Mutation
#TPX
#K-tournament selection
#fitness sharing selection
def island3(distanceMatrix, population, iters, lambd, mu, k, alpha, numCities, nns):
    alpha = 0.4
    fast_swap_lso(distanceMatrix, population)
    for i in range(iters):
        it_start = time.time()

        #Create offspring
        offspring = np.empty((mu, numCities), dtype = int)
        num_parents = 2*mu

        ##### SELECTION
        fitness_values = np.array([fitness(ind, distanceMatrix) for ind in population])
        #fitness_values = compute_all_shared_fitnesses(population, distanceMatrix)
        #selected_individuals = exp_selection(distanceMatrix, population, lambd, num_parents, fitness_values, selection_pressure) #Version WITH geometric decay
        selected_individuals = k_tournament_selection(population, num_parents, fitness_values, k)
            
        # Select from the population:
        for j in range(mu):
            p1 = selected_individuals[2*j]
            p2 = selected_individuals[2*j + 1]
            ##### RECOMBINATION 
            offspring[j] = tpx(p1, p2, alpha, alpha)
        SAM(offspring, alpha, nns)
        SAM(population, alpha, nns)
        fast_swap_lso(distanceMatrix, population)
        fast_swap_lso(distanceMatrix, offspring)

        population = elimination(distanceMatrix, population, offspring, lambd)
        it_end = time.time()
        print(i, "Island3 time:", it_end-it_start)

    return population









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
	n = dmatrix.shape[1]
	distance = 0
	for i in range(0,n-1):
		distance += dmatrix[individual[i], individual[i+1]]  
	distance += dmatrix[individual[n-1], individual[0]]      
	return distance

#def fitnessv2(individual, dmatrix):


@nb.njit()
def fitness2(individual, dmatrix):
	n = individual.shape[0]
	distance = 0
	for i in range(0,n-1):
		distance += dmatrix[individual[i], individual[i+1]]  
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
    alpha = 0.8
    sigma =  (n-1) * 0.2 #We need a distance function!

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


@nb.njit()
def IRGIBNNM(inds, a, nns):
    n = inds.shape[1]
    for k in range(inds.shape[0]):
        if np.random.uniform() < a:
            #First perform inv mutation
            i = np.random.randint(0,n-1)
            j = np.random.randint(i+1,n)
            inds[k][i:j] = np.flip(inds[k][i:j])

            #Insert random city (rci) close (-+5) from its nn
            rcidx = np.random.randint(0,n)
            rci = inds[k, rcidx]
            nn = nns[rci]

            nnidx = np.where(inds[k]==nn)[0][0]
            #If random city is left of nn, get -1, otherwise +1
            offsetSide = np.sign(rcidx - nnidx)
            swapCityIndex = (nnidx + offsetSide) % n
            if offsetSide > 0:
                inds[k][nnidx+2:rcidx+1] = inds[k][nnidx+1:rcidx]
                inds[k, swapCityIndex] = rci
            else:
                inds[k][rcidx:nnidx] = inds[k][rcidx+1:nnidx+1]
                inds[k, swapCityIndex] = rci


@nb.njit()
def RGIBNNM(inds, a, nns):
    n = inds.shape[1]
    for k in range(inds.shape[0]):
        if np.random.uniform() < a:
            #Insert random city (rci) close (-+5) from its nn
            rcidx = np.random.randint(0,n)
            rci = inds[k, rcidx]
            nn = nns[rci]

            nnidx = np.where(inds[k]==nn)[0][0]
            #If random city is left of nn, get -1, otherwise +1
            offsetSide = np.sign(rcidx - nnidx)
            swapCityIndex = (nnidx + offsetSide) % n
            if offsetSide > 0:
                inds[k][nnidx+2:rcidx+1] = inds[k][nnidx+1:rcidx]
                inds[k, swapCityIndex] = rci
            else:
                inds[k][rcidx:nnidx] = inds[k][rcidx+1:nnidx+1]
                inds[k, swapCityIndex] = rci


@nb.njit()
def SAM(inds, a, nns):
    choice = np.random.randint(4)
    if choice == 0:
        swapMutation(inds, a)
    elif choice == 1:
        invMutation(inds, a)
    elif choice == 2:
        RGIBNNM(inds, a, nns)
    elif choice == 3:
        IRGIBNNM(inds, a, nns)


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


@nb.njit()
def random_less_inf_gen(distance_matrix, lam):
    num_cities = len(distance_matrix[0])
    initial_population = np.empty((lam, num_cities), dtype=np.int64)
    for route in range(lam):
        city_status = np.ones(num_cities, dtype=np.bool_)
        first_city = np.random.randint(num_cities)
        prev_city = first_city
        initial_population[route, 0] = first_city
        city_status[first_city] = False
        for j in range(1, num_cities):
            unvisited_cities = np.nonzero(city_status)[0]
            next_city = choose_candidate(distance_matrix, unvisited_cities, 10, prev_city)
            prev_city = next_city
            initial_population[route, j] = next_city
            city_status[next_city] = False
    return initial_population    

@nb.njit()
def choose_candidate(dmatrix, candidates, n, prev):
    bestCand = -1
    bestVal = np.inf
    for i in range(n):
        candidate = np.random.choice(candidates)
        candidateVal = dmatrix[prev, candidate]
        if candidateVal <= bestVal:
            bestCand = candidate
            bestVal = candidateVal
    return bestCand


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
def apply_lso_iters(lso, dmatrix, pop, iters):
    for i in range(iters):
        lso(dmatrix, pop)

@nb.njit()
def apply_single_lso_iters(lso, dmatrix, ind, iters):
    curr_ind = ind
    for i in range(iters):
        curr_ind = lso(dmatrix, curr_ind)
    return curr_ind

@nb.njit()
def swap_lso(dmatrix, ind):
    curr_route = ind
    bestFit = fitness(curr_route, dmatrix)
    bestj = -1
    bestk = -1
    for j in range(ind.shape[0]-1):
        for k in range(j+1, ind.shape[0]):
            temp = curr_route[j]
            curr_route[j] = curr_route[k]
            curr_route[k] = temp
            fit = fitness(curr_route, dmatrix)
            if fit < bestFit:
                bestFit = fit
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
    bestInd = ind.copy()
    #print("fit of ind before", fitness(ind, dmatrix))
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
                    bestInd = ind.copy()
                #Perm2 102
                ind[i] = jval
                ind[j] = ival
                ind[k] = kval
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()
                #Perm3 120
                ind[i] = jval
                ind[j] = kval
                ind[k] = ival
                fit = fitness(ind, dmatrix) 
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()
                #Perm4 201
                ind[i] = kval
                ind[j] = ival
                ind[k] = jval    
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()   
                #Perm5 210
                ind[i] = kval
                ind[j] = jval
                ind[k] = ival
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()
                #Reset individual
                ind[i] = ival
                ind[j] = jval
                ind[k] = kval
    return bestInd
    #Perform best 3-swap
    #print("best found fit:", bestFit)
    #print("fit of result individual:", fitness(bestInd, dmatrix))




@nb.njit()
def fast_swap3_lso(dmatrix, ind):
    n = ind.shape[0]
    bestInd = ind.copy()
    #print("fit of ind before", fitness(ind, dmatrix))
    #i = 0, j = 1, k = 2
    bestFit = fitness(ind, dmatrix)
    iters = 2000
    for iter in range(iters):
        i = np.random.randint(n-2)
        j = np.random.randint(i+1, n-1)
        k = np.random.randint(j+1, n)

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
            bestInd = ind.copy()
        #Perm2 102
        ind[i] = jval
        ind[j] = ival
        ind[k] = kval
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        #Perm3 120
        ind[i] = jval
        ind[j] = kval
        ind[k] = ival
        fit = fitness(ind, dmatrix) 
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        #Perm4 201
        ind[i] = kval
        ind[j] = ival
        ind[k] = jval    
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()   
        #Perm5 210
        ind[i] = kval
        ind[j] = jval
        ind[k] = ival
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        #Reset individual
        ind[i] = ival
        ind[j] = jval
        ind[k] = kval
    return bestInd


@nb.njit()
def fast_swap_lso(dmatrix, pop):
    n = pop.shape[1]
    for i in range(pop.shape[0]):
        curr_route = pop[i]
        bestDiff = 0
        bestj = -1
        bestk = -1
        iters = 5000
        for j in range(iters):
            idi = np.random.randint(n-1)
            idj = np.random.randint(idi+1, n)
            slicei = max(0, idi-1)
            slicej = idj-1
            if slicej == idi:
                slicej = idj
            total = fitness2(curr_route[slicei:idi+2], dmatrix)
            total += fitness2(curr_route[slicej:idj+2], dmatrix)
            total += dmatrix[curr_route[n-1], curr_route[0]]
            temp = curr_route[idi]
            curr_route[idi] = curr_route[idj]
            curr_route[idj] = temp
            total -= fitness2(curr_route[slicei:idi+2], dmatrix)
            total -= fitness2(curr_route[slicej:idj+2], dmatrix)
            total -= dmatrix[curr_route[n-1], curr_route[0]]

            if total > bestDiff:
                bestDiff = total
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
def fast_swap3_lso_pop(dmatrix, pop):
    n = pop.shape[1]
    for id in range(pop.shape[0]):
        bestInd = pop[id].copy()
        ind = pop[id].copy()
        #print("fit of ind before", fitness(ind, dmatrix))
        #i = 0, j = 1, k = 2
        bestDiff = 0
        iters = 500
        for iter in range(iters):
            i = np.random.randint(n-2)
            j = np.random.randint(i+1, n-1)
            k = np.random.randint(j+1, n)

            slicei = max(0, i-1)
            slicej = j-1
            if slicej == i:
                slicej = j
            sliceklow = k-1
            if sliceklow == j:
                sliceklow = k

            initialTotal = computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            ival = ind[i]
            jval = ind[j]
            kval = ind[k]               
            #Perm1 021
            ind[i] = ival
            ind[j] = kval
            ind[k] = jval
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()
                
            #Perm2 102
            ind[i] = jval
            ind[j] = ival
            ind[k] = kval
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()
                
            #Perm3 120
            ind[i] = jval
            ind[j] = kval
            ind[k] = ival
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()

            #Perm4 201
            ind[i] = kval
            ind[j] = ival
            ind[k] = jval    
            fit = fitness(ind, dmatrix)
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()
            #Perm5 210
            ind[i] = kval
            ind[j] = jval
            ind[k] = ival
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()
            #Reset individual
            ind[i] = ival
            ind[j] = jval
            ind[k] = kval
            if bestDiff > 0:
                break
        pop[id] = bestInd

@nb.njit
def computeTotal(dmatrix, ind, i, slicei,j, slicej, sliceklow, k):
    n = ind.shape[0]
    total = 0
    total += fitness2(ind[slicei:i+2], dmatrix)
    total += fitness2(ind[slicej:j+2], dmatrix)
    total += fitness2(ind[sliceklow:k+2], dmatrix)
    total += dmatrix[ind[n-1], ind[0]]
    return total

@nb.njit()
def opt2_lso(dmatrix, ind):
    curr_route = ind
    bestFit = fitness(curr_route, dmatrix)
    bestDiff = 0

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
    return ind


@nb.njit()
def fast_opt2_lso(dmatrix, pop):
    n = pop.shape[1]
    for k in range(pop.shape[0]):
        curr_route = pop[k]
        # f1 = fitness(curr_route, dmatrix)
        bestDiff = 0
        bestj = -1
        bestk = -1
        iters = 1000
        for a in range(iters):
            i = np.random.randint(0,n)
            j = np.random.randint(i+1,n+1)
            slicei = max(0, i-1)
            total = fitness2(curr_route[slicei:j+1], dmatrix)
            total += dmatrix[curr_route[n-1], curr_route[0]]

            pop[k][i:j] = np.flip(pop[k][i:j])

            total -= fitness2(curr_route[slicei:j+1], dmatrix)
            total -= dmatrix[curr_route[n-1], curr_route[0]]
            if total > bestDiff:
                bestDiff = total
                bestj = i
                bestk = j
            #Undo operation
            pop[k][i:j] = np.flip(pop[k][i:j])
        pop[k][bestj:bestk] = np.flip(pop[k][bestj:bestk])
        # f2 = fitness(pop[k], dmatrix)
        # print("f1:", f1)
        # print("f2:", f2)



@nb.njit()
def fast_opt3_lso(dmatrix, ind):
    n = ind.shape[0]
    bestInd = ind.copy()
    #i = 0, j = 1, k = 2
    bestFit = fitness(ind, dmatrix)
    #Maintain a gap of 2 between indices, as flip does not do anything otherwise
    iters = 2000
    for iter in range(iters):
        i = np.random.randint(n-3)
        j = np.random.randint(i+2, n-1)
        k = np.random.randint(j+2, n+1)

        #Perm1 i,k
        ind[i:k] = np.flip(ind[i:k])
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        ind[i:k] = np.flip(ind[i:k])

        #Perm2 j,k
        ind[j:k] = np.flip(ind[j:k])
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        ind[j:k] = np.flip(ind[j:k])

        #Perm3 i,j
        ind[i:j] = np.flip(ind[i:j])
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        ind[i:j] = np.flip(ind[i:j])

        #Perm4 ij, jk
        ind[i:j] = np.flip(ind[i:j])
        ind[j:k] = np.flip(ind[j:k])
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        ind[j:k] = np.flip(ind[j:k])
        ind[i:j] = np.flip(ind[i:j])

        #Perm5 ik, jk
        ind[i:k] = np.flip(ind[i:k])
        ind[j:k] = np.flip(ind[j:k])
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        ind[j:k] = np.flip(ind[j:k])
        ind[i:k] = np.flip(ind[i:k])
        #Perm6 ik, ij
        ind[i:k] = np.flip(ind[i:k])
        ind[i:j] = np.flip(ind[i:j])
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        ind[i:j] = np.flip(ind[i:j])
        ind[i:k] = np.flip(ind[i:k])
        
        #Perm7 ik, ij, jk
        ind[i:k] = np.flip(ind[i:k])
        ind[i:j] = np.flip(ind[i:j])
        ind[j:k] = np.flip(ind[j:k])
        fit = fitness(ind, dmatrix)
        if fit < bestFit:
            bestFit = fit
            bestInd = ind.copy()
        ind[j:k] = np.flip(ind[j:k])
        ind[i:j] = np.flip(ind[i:j])
        ind[i:k] = np.flip(ind[i:k])
    return bestInd

@nb.njit()
def faster_opt3_lso(dmatrix, pop):
    n = pop.shape[1]
    for id in range(pop.shape[0]):
        bestInd = pop[id].copy()
        ind = pop[id].copy()
        #i = 0, j = 1, k = 2
        bestFit = fitness(ind, dmatrix)
        origFit = bestFit
        #Maintain a gap of 2 between indices, as flip does not do anything otherwise
        iters = 400
        for iter in range(iters):
            i = np.random.randint(n-3)
            j = np.random.randint(i+2, n-1)
            k = np.random.randint(j+2, n+1)

            #Perm1 i,k
            ind[i:k] = np.flip(ind[i:k])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
            ind[i:k] = np.flip(ind[i:k])

            #Perm2 j,k
            ind[j:k] = np.flip(ind[j:k])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
            ind[j:k] = np.flip(ind[j:k])

            #Perm3 i,j
            ind[i:j] = np.flip(ind[i:j])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
            ind[i:j] = np.flip(ind[i:j])

            #Perm4 ij, jk
            ind[i:j] = np.flip(ind[i:j])
            ind[j:k] = np.flip(ind[j:k])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
            ind[j:k] = np.flip(ind[j:k])
            ind[i:j] = np.flip(ind[i:j])

            #Perm5 ik, jk
            ind[i:k] = np.flip(ind[i:k])
            ind[j:k] = np.flip(ind[j:k])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
            ind[j:k] = np.flip(ind[j:k])
            ind[i:k] = np.flip(ind[i:k])
            #Perm6 ik, ij
            ind[i:k] = np.flip(ind[i:k])
            ind[i:j] = np.flip(ind[i:j])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
            ind[i:j] = np.flip(ind[i:j])
            ind[i:k] = np.flip(ind[i:k])
            
            #Perm7 ik, ij, jk
            ind[i:k] = np.flip(ind[i:k])
            ind[i:j] = np.flip(ind[i:j])
            ind[j:k] = np.flip(ind[j:k])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
            ind[j:k] = np.flip(ind[j:k])
            ind[i:j] = np.flip(ind[i:j])
            ind[i:k] = np.flip(ind[i:k])

            if bestFit < origFit:
                break
        pop[i] = bestInd


######################## Mutation Rate Related Stuff #########################
def combineAlphas(a1, a2):
    b = 2 * np.random.uniform() - 0.5 #Between -0.5 and 1.5
    a = a1 + b * (a2-a1) 
    return np.abs(a)


######################## MISC HELPER FUNCTIONS ############################

#At some point we need to translate our path into a cycle
def solutionToCycle(path):
    return []



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


testArray1 = np.array([[0, 1.5, 4.8, 3.4],
                       [2.8, 0, 6.4, 8.7],
					   [10., 5.4, 0, 1.],
                       [100., 3.8, 9.3, 0]])

testArray2 = np.array([[0, np.inf, 4.8, 3.4],
                       [2.8, 0, 6.4, 8.7],
					   [np.inf, 5.4, 0, 1.],
                       [10., 3.8, np.inf, 0]])
#population = random_less_inf_gen(testArray2, 10)
#print("population:", population)
population = np.array([[0,1,2,3],[3,1,2,0], [2,3,0,1]])
#fast_swap3_lso_pop(testArray1, population)
#print("population", population)
#a = np.array([0,1,2,3])
#dmatrixmod = testArray1.copy()
#np.fill_diagonal(dmatrixmod, np.inf)
#nns = np.argmin(dmatrixmod, axis=1)
#print("nns", nns)
#IRGIBNNM(population, 1, nns)
#optimizeBestInd(swap3_lso, a, testArray1)
#swap3_lso(testArray1, a)
#opt3_lso(testArray1, a)
# print("pop:", population)
# swap_lso(testArray1, population)
# print("pop:", population)

#print("dists", dist_to_pop(a,c))

prog = TspProg()
params = Parameters(lambd=120, mu=120, k=5, its=1000)
prog.optimize("tour1000.csv", params)
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
