import Reporter
import numpy as np
import random
from scipy.stats import truncexpon
import math
import matplotlib.pyplot as plt
import time
import numba as nb

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
        a = 0.9
        selection_pressure = a

        bestInd = np.random.permutation(numCities)
        
        ##### GENERATION
        


        #population = random_generation(distanceMatrix, lam)
        population = nn_krandom_generation(distanceMatrix, lam)
        #routes = parallel_diversification_generation(distanceMatrix, lam)
        #routes = sequential_diversification_generation(distanceMatrix, lam)
        for i in range(lam):
            start = time.time()
            #compute_all_shared_fitnesses(population, distanceMatrix)
            route = population[i]
            dist_to_pop2(route, population)
            end = time.time()
            print("Time:", end - start)


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
            #selected_individuals = exp_selection(distanceMatrix, population, lam, num_parents) #Version WITHOUT geometric decay
            #selected_individuals = exp_selection(distanceMatrix, population, lam, num_parents, selection_pressure) #Version WITH geometric decay
            selected_individuals = k_tournament_selection(distanceMatrix, population, num_parents, k)
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

			##### ELIMINATION
            population = elimination(distanceMatrix, population, offspring, lam)


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
                break
            
        print("Route of best individual:")
        printIndividual(bestInd, distanceMatrix)
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
def compute_all_shared_fitnesses(population, dmatrix):
    n = population.shape[0]
    fitnesses = np.empty(n)
    for i in range(n):
        route = population[i]
        fitnesses[i] = shared_fitness(route, dmatrix, population)
    return fitnesses



@nb.njit()
def shared_fitness(individual, dmatrix, population, betaInit=0):
    n = individual.shape[0]
    alpha = 1
    sigma =  (n-1) * 0.2 #We need a distance function!

    distances = dist_to_pop(individual, population)
    beta = betaInit
    for i in range(n):
        dist = distances[i]
        if dist <= sigma:
            beta += 1 - (dist/sigma)**alpha
    origFit = fitness(individual, dmatrix)
    res = origFit * beta**np.sign(origFit)
    return res


@nb.njit()
def dist_to_pop(route, pop):
    popSize = pop.shape[0]
    output = np.zeros(popSize, dtype=np.int64)
    for i in range(popSize):
        output[i] = common_edges_dist2(route, pop[i])

    return output

@nb.njit()
def dist_to_pop2(route, pop):
    popSize = pop.shape[0]
    output = np.zeros(popSize, dtype=np.int64)
    n = pop.shape[1]
    edges = np.empty(n*n)
    for i in range(popSize):
        output[i] = common_edges_dist20(route, pop[i],edges, i)

    return output

#An edge might be 4-5 OR 5-4...
#I hash the pairs essentially, make lookups cheaper, cheaper to store ints than tuples
@nb.njit()
def common_edges_dist2(route1, route2):
    num_edges = 0
    n = route1.shape[0]
    edges = []
    for i in range(n-1):
        a = route1[i]
        b = route1[i+1]
        if a < b:
            val = a * 1000 + b
            edges.append(val)
        else:
            val = b * 1000 + a
            edges.append(val)
    if route1[n-1] < route1[0]:
        edges.append(route1[n-1]*1000+route1[0])
    else:
        edges.append(route1[0]*1000+route1[n-1])

    edges = set(edges)
    for i in range(n-1):
        a = route2[i]
        b = route2[i+1]
        if a < b:            
            val = a * 1000 + b
        else:
            val = b * 1000 + a
        if val in edges:
            num_edges += 1
    val1 = route2[n-1] * 1000 + route2[0]
    val2 = route2[0] * 1000 + route2[n-1]
    if val1 in edges or val2 in edges:
       num_edges += 1
    return n-num_edges




@nb.njit()
def common_edges_dist20(route1, route2, edges, num):
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
    b = route2[b]
    if a < b:            
        val = a * 1000 + b
    else:
        val = b * 1000 + a
    if edges[val] == num:
        num_edges += 1
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



### PMX ###
def pmx(candidate11, candidate22, a1, a2):
    candidate1 = list(candidate11)
    candidate2 = list(candidate22)
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
    return np.array(offspring)
### END PMX


def recursive_fill(index, index_set, item, offspring, candidate):
    if index not in index_set:
        offspring[index] = item
        index_set.add(index)
    else:
        new_item = offspring[index]
        new_index = candidate.index(new_item)
        recursive_fill(new_index, index_set, item, offspring, candidate)


#Faster for bigger arrays
def pmx2(candidate1, candidate2, a1, a2):
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
    for i in loop_indexes:
        curr_city = candidate2[i]
        while curr_city in interval:
            new_index = index_map[curr_city]
            curr_city = candidate2[new_index]
        offspring[i] = curr_city

    #print("offspring:", offspring)
    return offspring


def pmx2_loop(candidate1, candidate2, a1, a2):
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
    offspring = pmx2loop(index_map, candidate2, offspring, loop_indexes, interval)

    #print("offspring:", offspring)
    return offspring
@nb.njit()
def pmx2loop(index_map, candidate2, offspring, loop_indexes, interval):
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

    fitness_values = np.array([fitness(ind, distance_matrix) for ind in population])

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


def exp_selection(dmatrix, pop, l, mu, selection_pressure=0.01):
    # Create the distribution:
    a = math.log(selection_pressure)/(l-1)
    beta = -1/a
    X = truncexpon(b=(l)/beta, loc=0, scale=beta)
    data = X.rvs(mu)
    index = [int(a) for a in np.floor(data)]
    
    pop_fitness = np.apply_along_axis(fitness, 1, pop, dmatrix=dmatrix)
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



def swap_lso(dmatrix, pop):
    for i in range(pop.shape[0]):
        curr_route = pop[i]
        bestFit = fitness()

def swap_single(dmatrix, ind, pop):
    bestFit = fitness(ind)


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

# testArray1 = np.array([[0, 1.5, 2.4, 3.4],
#                        [2.8, 0, 5.1, 1.3],
# 					   [10., 5.4, 0, 9.5],
#                        [6.6, 3.8, 9.3, 0]])

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
# print("dist:", common_edges_dist2(a, b))
# edges = np.empty(81)
# print("dist:", common_edges_dist20(a, b, edges, 1))



#print("dists", dist_to_pop(a,c))

prog = TspProg()
params = Parameters(lambd=1000, mu=1000, k=5, its=2000)
prog.optimize("tour1000.csv", params)
