import Reporter
import numpy as np
import random
from scipy.stats import truncexpon
import math
import matplotlib.pyplot as plt
import time
import numba as nb
import multiprocessing as mp


# Modify the class name to match your student number.
class R0975929:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
    def optimize(self, filename, testFile = None):
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
        numCities = len(distanceMatrix[0])
        params = Parameters(lambd=600, mu=600, k=5, its=1000)
        kmig = 5
        exchangeRate = 3

        if numCities == 50:
            print("50 Cities")
            params = Parameters(lambd=800, mu=800, k=5, its=1000)
            exchangeRate = 10
        if numCities == 100:
            print("100 Cities")
            params = Parameters(lambd=440, mu=440, k=5, its=1000)
            exchangeRate = 5
        if numCities == 200:
            print("200 Cities")
            params = Parameters(lambd=360, mu=360, k=5, its=1000)
            exchangeRate = 5
        if numCities == 500:
            print("500 Cities")
            params = Parameters(lambd=180, mu=180, k=5, its=1000)
        if numCities == 750:
            print("750 Cities")
            params = Parameters(lambd=160, mu=160, k=5, its=1000)
        if numCities == 1000:
            print("1000 Cities")
            params = Parameters(lambd=120, mu=120, k=5, its=1000)

        lam = params.la #Population size
        mu = params.mu  #Offspring size  
        k = params.k    #K-tournament selection param  
        iterations = params.iterations
        alpha = 0.2
        print("Lambda:", lam, "Mu:", mu)
        bestInd = np.random.permutation(numCities)
        bestIndFit = np.inf
        ##### GENERATION
        islandIters = 5

        islandSize = int(lam/4)
        islandmu = int(mu/4)

        island1Size = int(lam/8)
        island1mu = int(lam/8)
        island2pressure = 0.4



        dmatrixmod = distanceMatrix.copy()
        np.fill_diagonal(dmatrixmod, np.inf)
        nns = np.argmin(dmatrixmod, axis=1)



        print("Initializing island populations")
        island1pop = nn_krandom_generation(distanceMatrix, island1Size)
        #island1pop = random_less_inf_gen(distanceMatrix, island1Size)
        #fast_opt2_lso_exh(distanceMatrix, island1pop)
        fast_swap3_lso_pop_exh(distanceMatrix, island1pop)

        island2pop = nn_krandom_generation(distanceMatrix, islandSize)
        # island2pop = random_less_inf_gen(distanceMatrix, islandSize)
        RGIBNNM(island2pop, 0.5, nns)
        invMutation(island2pop, 0.5)
        # island2pop = random_generation(distanceMatrix, islandSize)

        # faster_opt3_lso_exh(distanceMatrix, island2pop)
        island3pop = nkn_krandom_generation(distanceMatrix, islandSize, 2)
        #island3pop = random_less_inf_gen(distanceMatrix, islandSize)

        faster_opt3_lso_exh(distanceMatrix, island3pop)
        island4pop = nn_krandom_generation(distanceMatrix, islandSize)

    
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
            island2pop, island2pressure = island2(distanceMatrix, island2pop, islandIters, islandSize, islandmu, island2pressure, alpha, numCities, nns)
            island3pop = island3(distanceMatrix, island3pop, islandIters, islandSize, islandmu, k, alpha, numCities, nns)
            island4pop = island4(distanceMatrix, island4pop, islandIters, islandSize, islandmu, k, alpha, numCities, nns)

            evalPop(island1pop, distanceMatrix, "Island1:")
            evalPop(island2pop, distanceMatrix, "Island2:")
            evalPop(island3pop, distanceMatrix, "Island3:")
            evalPop(island4pop, distanceMatrix, "Island4:")

            # #Ring topology
            island1candidateVals = compute_all_shared_fitnesses_island(island1pop, island2pop, distanceMatrix)
            island2candidateVals = compute_all_shared_fitnesses_island(island2pop, island3pop, distanceMatrix)
            island3candidateVals = compute_all_shared_fitnesses_island(island3pop, island4pop, distanceMatrix)
            island4candidateVals = compute_all_shared_fitnesses_island(island4pop, island1pop, distanceMatrix)

            i1migrators, i1indices = k_tournament_migration(island1pop, exchangeRate, island1candidateVals, kmig)
            i2migrators, i2indices = k_tournament_migration(island2pop, exchangeRate, island2candidateVals, kmig)
            i3migrators, i3indices = k_tournament_migration(island3pop, exchangeRate, island3candidateVals, kmig)
            i4migrators, i4indices = k_tournament_migration(island4pop, exchangeRate, island4candidateVals, kmig)

            island1pop[i1indices, :] = i4migrators
            island2pop[i2indices, :] = i1migrators
            island3pop[i3indices, :] = i2migrators
            island4pop[i4indices, :] = i3migrators

            
            bestInd1 = optimizeBestInd(fast_opt3_lso_ind_exh, island1pop, distanceMatrix)
            bestInd2 = optimizeBestInd(fast_swap3_ind_exh, island2pop, distanceMatrix)
            bestInd3 = optimizeBestInd(fast_opt3_lso_ind_exh, island3pop, distanceMatrix)
            bestInd4 = optimizeBestInd(fast_swap3_ind_exh, island4pop, distanceMatrix)


            i1s = worstIndsArgs(island1pop, distanceMatrix, 1)
            i2s = worstIndsArgs(island2pop, distanceMatrix, 1)
            i3s = worstIndsArgs(island3pop, distanceMatrix, 1)
            i4s = worstIndsArgs(island4pop, distanceMatrix, 1)

            island1pop[i1s,:] = bestInd4
            island2pop[i2s,:] = bestInd1
            island3pop[i3s,:] = bestInd2
            island4pop[i4s,:] = bestInd3

            population = np.vstack((island1pop, island2pop, island3pop, island4pop))

			##### EVALUATION
            objectiveValues = np.array([fitness(ind, distanceMatrix) for ind in population])
            bestIndIdx = np.argmin(objectiveValues)

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
            bestSolution = solutionToCycle(population[np.argmin(objectiveValues)])
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            it_end = time.time()
            print("Iteration time:", it_end-it_start)
            if objectiveValues[bestIndIdx] < bestIndFit:
                bestInd = population[bestIndIdx]
                bestIndFit = objectiveValues[bestIndIdx]   

            if timeLeft < 0 or i >= iterations:
                print("No time left, stopping!")
                break
            
        #printIndividual(bestInd, distanceMatrix)
        print("Final mean:", meanHist[len(meanHist)-1], ", Final best:", minimumHist[len(minimumHist)-1])
        plotResuts(meanHist, minimumHist)
		# Your code here.
        return 0
    
def solutionToCycle(path):
    id = np.where(path == 0)[0][0]
    return np.roll(path, -id)

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
# alpha = 0.5
# Initial faster_opt3_lso_exh
# K tournament (fitness sharing)
# PMX
# IRGIBNMM and RGIBNMM
# 5x fast_swap3_lso_pop on pop
#fast_swap_lso on offspring
def island1(distanceMatrix, population, iters, lambd, mu, k, alpha, numCities, nns):
    alpha = 0.5
    faster_opt3_lso_exh(distanceMatrix, population)
    for i in range(iters):
        it_start = time.time()

        #Create offspring
        offspring = np.empty((mu, numCities), dtype = int)
        num_parents = 2*mu

        ##### SELECTION
        #fitness_values = np.array([fitness(ind, distanceMatrix) for ind in population])
        fitness_values = compute_all_shared_fitnesses(population, distanceMatrix)
        k = 3
        selected_individuals = k_tournament_selection(population, num_parents, fitness_values, k)
        # Select from the population:
        for j in range(mu):
            p1 = selected_individuals[2*j]
            p2 = selected_individuals[2*j + 1]
            ##### RECOMBINATION 
            offspring1 = pmx(p1, p2, alpha, alpha)
            offspring2 = pmx(p2, p1, alpha, alpha)
            if fitness(offspring1, distanceMatrix) < fitness(offspring2, distanceMatrix):
                offspring[j] = offspring1
            else:
                offspring[j] = offspring2
        IRGIBNNM(offspring, alpha, nns)        
        RGIBNNM(population, alpha, nns)
        for _ in range(5):
            fast_swap3_lso_pop(distanceMatrix, population)
            fast_swap3_lso_pop(distanceMatrix, offspring)

        #population = elimination(distanceMatrix, population, offspring, lambd)
        shared_fitness_elimination(distanceMatrix, population, offspring, lambd, sigmaMult = 0.1, alpha=0.5)

        it_end = time.time()
        print(i, "Island1 time:", it_end-it_start)


    return population


#Island with:
#Initial fast_swap3_lso_pop_exh
#Exp_selection (fitness sharing)
#TPX
#Inverse Mutation
# 5x fast_opt2_lso
def island2(distanceMatrix, population, iters, lambd, mu, selection_pressure, alpha, numCities, nns):
    alpha = 0.2
    a = 0.99
    fast_swap3_lso_pop_exh(distanceMatrix, population)
    for i in range(iters):
        it_start = time.time()

        #Create offspring
        offspring = np.empty((mu, numCities), dtype = int)
        num_parents = 2*mu

        ##### SELECTION
        #fitness_vals_for_use = np.apply_along_axis(fitness, 1, population, dmatrix=distanceMatrix)
        fitness_values = compute_all_shared_fitnesses(population, distanceMatrix, 0.4, 1)
        selected_individuals = exp_selection(distanceMatrix, population, lambd, num_parents, fitness_values, selection_pressure) #Version WITH geometric decay
        #selected_individuals = k_tournament_selection(population, num_parents, fitness_values, 5)
        if i % 2 == 0 and selection_pressure > 0.01: #Set how aggressive the decay should be
            selection_pressure *= a

        # Select from the population:
        for j in range(mu):
            p1 = selected_individuals[2*j]
            p2 = selected_individuals[2*j + 1]
            ##### RECOMBINATION 
            offspring[j] = pmx(p1, p2, alpha, alpha)
        invMutation(offspring, alpha)
        invMutation(population, alpha)

        faster_opt3_lso(distanceMatrix, offspring)
        fast_swap3_lso_pop(distanceMatrix, population)

        population = elimination(distanceMatrix, population, offspring, lambd)
        #shared_fitness_elimination(dmatrix, population, offspring, lambd, sigmaMult = 0.3, alpha=0.5)

        it_end = time.time()
        print(i, "Island2 time:", it_end-it_start)

    return population, selection_pressure



#Island with:
#Inverse Mutation
#TPX
#K-tournament selection
#fitness sharing selection
def island3(distanceMatrix, population, iters, lambd, mu, k, alpha, numCities, nns):
    alpha = 0.3
    faster_opt3_lso_exh(distanceMatrix, population)
    for i in range(iters):
        it_start = time.time()

        #Create offspring
        offspring = np.empty((mu, numCities), dtype = int)
        num_parents = 2*mu

        ##### SELECTION
        fitness_values = compute_all_shared_fitnesses(population, distanceMatrix)
        selected_individuals = k_tournament_selection(population, num_parents, fitness_values, 3)

        # Select from the population:
        for j in range(mu):
            p1 = selected_individuals[2*j]
            p2 = selected_individuals[2*j + 1]
            ##### RECOMBINATION 
            offspring1 = ox(p1, p2, alpha, alpha)
            offspring2 = ox(p2, p1, alpha, alpha)
            if fitness(offspring1, distanceMatrix) < fitness(offspring2, distanceMatrix):
                offspring[j] = offspring1
            else:
                offspring[j] = offspring2
        swapMutation(offspring, alpha)        
        swapMutation(population, alpha)
        for _ in range(5):
            fast_swap3_lso_pop(distanceMatrix, population)
        fast_swap_lso(distanceMatrix, offspring)

        population = elimination(distanceMatrix, population, offspring, lambd)
        #shared_fitness_elimination(dmatrix, population, offspring, lambd, sigmaMult = 0.3, alpha=0.5)

        it_end = time.time()
        print(i, "Island3 time:", it_end-it_start)

    return population


def island4(distanceMatrix, population, iters, lambd, mu, selection_pressure, alpha, numCities, nns):
    alpha = 0.2
    fast_swap3_lso_pop_exh(distanceMatrix, population)
    orOptAll(distanceMatrix, population)
    #full_swap3_lso_pop(distanceMatrix, population)
    for i in range(iters):
        it_start = time.time()

        #Create offspring
        offspring = np.empty((mu, numCities), dtype = int)
        num_parents = 2*mu

        ##### SELECTION
        fitness_values = compute_all_shared_fitnesses(population, distanceMatrix)
        selected_individuals = k_tournament_selection(population, num_parents, fitness_values, 5)

        # Select from the population:
        for j in range(mu):
            p1 = selected_individuals[2*j]
            p2 = selected_individuals[2*j + 1]
            f1 = fitness(p1, distanceMatrix)
            f2 = fitness(p2, distanceMatrix)
            if f1 == f2:
                p2 = invMutationI(p2, 1)
            ##### RECOMBINATION 
            offspring1 = scx(p1, p2, distanceMatrix, alpha, alpha)
            offspring2 = scx(p2, p1, distanceMatrix, alpha, alpha)
            if fitness(offspring1, distanceMatrix) < fitness(offspring2, distanceMatrix):
                offspring[j] = offspring1
            else:
                offspring[j] = offspring2
        RGIBNNM(offspring, alpha, nns)
        RGIBNNM(population, alpha, nns)
        for _ in range(5):
            fast_swap3_lso_pop(distanceMatrix, offspring)
        fast_swap_lso(distanceMatrix, offspring)
        #orOptAll(distanceMatrix, population)
        population = elimination(distanceMatrix, population, offspring, lambd)
        it_end = time.time()
        print(i, "Island4 time:", it_end-it_start)

    return population


def k_tournament_migration(population, num_parents, fitness_values, k=5):

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
    
    if distance < 0:
        return np.inf
    return distance


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
def shared_fitness_elimination(dmatrix, population, offspring, num_survivors, sigmaMult=0.2,alpha=0.8):
    individuals = np.vstack((population, offspring))
    n = population.shape[1]
    survivors = np.empty((num_survivors, n), dtype=np.int64)
    num_survivors = survivors.shape[0]

    edges = np.zeros(1000000, dtype=np.int64)
    idx = compute_fitness_vals_best_id(individuals, None, dmatrix, edges, sigmaMult, alpha)
    survivors[0] = individuals[idx,:]

    for i in range(1, num_survivors):
        edges[:] = 0
        idx = compute_fitness_vals_best_id(individuals, survivors[0:i,:], dmatrix, edges, sigmaMult, alpha)
        survivors[i] = individuals[idx,:]
    return survivors



@nb.njit()
def compute_fitness_vals_best_id(individuals, survivors, dmatrix, edges, sigmaMult=0.2, alpha=0.8):
    num_individuals = individuals.shape[0]
    best_val = np.inf
    best_index = 0
    for j in range(num_individuals):
        fitness_val = shared_fitness(individuals[j], dmatrix, edges, survivors, 1, sigmaMult, alpha)
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
def compute_all_shared_fitnesses(population, dmatrix, sigmaMult=0.2, alpha=0.8):

    n = population.shape[0]
    edges = np.zeros(1000000, dtype=np.int64)
    fitnesses = np.empty(n)
    for i in range(n):
        route = population[i]
        fitnesses[i] = shared_fitness(route, dmatrix, edges, population, sigmaMult, alpha)
        edges[:] = 0
    return fitnesses



@nb.njit()
def shared_fitness(individual, dmatrix, edges, population=None, betaInit=0, sigmaMult=0.2, alpha=0.8):
    if population is None:
        return fitness(individual, dmatrix)

    n = individual.shape[0]
    alpha = alpha
    sigma =  (n-1) * sigmaMult #We need a distance function!

    distances = dist_to_pop(individual, population, edges)
    beta = betaInit
    for i in range(population.shape[0]):
        dist = distances[i]
        if dist <= sigma:
            beta += 1 - (dist/sigma)**alpha
    origFit = fitness(individual, dmatrix)
    res = origFit * beta**np.sign(origFit)
    if res == 0:
        return origFit
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


@nb.njit()
def invMutationI(indi, a):
    if np.random.uniform() < a:
        ind = indi.copy()
        i = np.random.randint(0,len(ind)-1)
        j = np.random.randint(i+1,len(ind))
        #individual.route[i:j] = individual.route[i:j][::-1]
        ind[i:j] = np.flip(ind[i:j])
    return ind

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

@nb.njit()
def scx(candidate1, candidate2, dmatrix, alpha1, alpha2, startIndex=0):
    n = candidate1.shape[0]
    city_indexes1 = np.empty(n, dtype=np.int64)
    city_indexes2 = np.empty(n, dtype=np.int64)
    city_lost1 = np.empty(n, dtype=np.bool_)
    city_lost2 = np.empty(n, dtype=np.bool_)

    for i in range(0, n):
        c1 = candidate1[i]
        c2 = candidate2[i]
        city_indexes1[c1] = i
        city_indexes2[c2] = i
        city_lost1[c1] = True
        city_lost2[c2] = True
    curr_city = candidate1[startIndex]
    updateCityStatus(curr_city, city_lost1)
    updateCityStatus(curr_city, city_lost2)

    offspring = np.empty(n, dtype = np.int64)
    offspring[0] = curr_city
    for i in range(1, n):
        c1 = findNeighbor(curr_city, candidate1, city_indexes1, city_lost1, n)
        c2 = findNeighbor(curr_city, candidate2, city_indexes2, city_lost2, n)
        dist1 = dmatrix[curr_city, c1]
        dist2 = dmatrix[curr_city, c2]
        #print("current city:", curr_city, "with neighbors:", c1, c2, "and dists", dist1, dist2)
        if dist1 <= dist2:
            offspring[i] = c1
            updateCityStatus(c1, city_lost1)
            updateCityStatus(c1, city_lost2)
            curr_city = c1
        else:
            offspring[i] = c2
            updateCityStatus(c2, city_lost1)
            updateCityStatus(c2, city_lost2)
            curr_city = c2
    return offspring
    
@nb.njit()
def findNeighbor(city, candidate, indexes, lostStatus, n):
    index= indexes[city]
    lost = False
    neigh = -1
    while not lost:
        index = (index + 1) % n
        neigh = candidate[index]
        lost = lostStatus[neigh]
    return neigh

@nb.njit()
def updateCityStatus(city, losts):
    losts[city] = False
    

@nb.njit()
def ox(candidate1, candidate2, alpha1, alpha2):
    length = candidate1.size
    # Choose our segment to add:
    a = np.random.randint(length-1)
    b = np.random.randint(a+1, length)
    offspring = np.zeros(length, dtype=np.int64) - 1
    offspring[a:b] = candidate1[a:b]
    toAvoid = set(candidate1[a:b])
    #Start insertion from end of newly inserted segment and wrap around
    insertionPoint = b
    for i in range(b, length):
        while offspring[insertionPoint] != -1 and insertionPoint < length:
            insertionPoint += 1
        curr_city = candidate2[i]
        if curr_city not in toAvoid:
            if insertionPoint == length:
                insertionPoint = 0
            offspring[insertionPoint] = curr_city
            insertionPoint += 1
    for i in range(b):
        while offspring[insertionPoint] != -1 and insertionPoint < length:
            insertionPoint += 1
        curr_city = candidate2[i]
        if curr_city not in toAvoid:
            if insertionPoint == length:
                insertionPoint = 0
            offspring[insertionPoint] = curr_city
            insertionPoint += 1
    return offspring


@nb.njit()
def ox_opt(candidate1, candidate2, alpha1, alpha2):
    length = candidate1.size
    # Choose our segment to add:
    seglength = np.random.randint(length-1)
    bestSeg = np.inf
    a = -1
    b = -1
    for i in range(0, length-seglength+1):
        segVal = fitness2(candidate1[i:seglength])
        if segVal <= bestSeg:
            bestSeg = segVal
            a = i
            b = i+seglength

    offspring = np.zeros(length, dtype=np.int64) - 1
    offspring[a:b] = candidate1[a:b]
    toAvoid = set(candidate1[a:b])
    #Start insertion from end of newly inserted segment and wrap around
    insertionPoint = b
    for i in range(b, length):
        while offspring[insertionPoint] != -1 and insertionPoint < length:
            insertionPoint += 1
        curr_city = candidate2[i]
        if curr_city not in toAvoid:
            if insertionPoint == length:
                insertionPoint = 0
            offspring[insertionPoint] = curr_city
            insertionPoint += 1
    for i in range(b):
        while offspring[insertionPoint] != -1 and insertionPoint < length:
            insertionPoint += 1
        curr_city = candidate2[i]
        if curr_city not in toAvoid:
            if insertionPoint == length:
                insertionPoint = 0
            offspring[insertionPoint] = curr_city
            insertionPoint += 1
    return offspring



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

        initial_population[new_route] = tour
    return initial_population



def nkn_krandom_generation(distance_matrix, lam, k):
    num_cities = len(distance_matrix[0])
    initial_population = np.empty((lam, num_cities), dtype=np.int64)

    for new_route in range(lam):
        #Changed it to a boolean filter to get around having to use .remove/.delete, much faster this way.
        unvisited_cities = np.ones(num_cities, dtype=np.bool_)

        tour = np.full(num_cities, -1)
        tour[0] = 0
        unvisited_cities[0] = False   
        for i in range(1, num_cities):
            if tour[i] != -1:
                continue
            dists = distance_matrix[tour[i-1], unvisited_cities]
            #print("dists:", dists)
            nearest_city_order = np.argsort(dists)
            num_left = dists.shape[0]
            k_pot = np.random.randint(0,k)
            real_k = min(num_left-1, k_pot)
            nearest_city_index = nearest_city_order[real_k]
            nearest_city = np.nonzero(unvisited_cities)[0][nearest_city_index]
            tour[i] = nearest_city
            unvisited_cities[nearest_city] = False

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
    n = ind.shape[0]
    curr_route = ind
    bestDiff = 0
    bestj = -1
    bestk = -1
    for idi in range(n-1):
        for idj in range(idi+1, n):
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
    temp = ind[bestj]
    ind[bestj] = ind[bestk]
    ind[bestk] = temp
    return ind



def fast_swap3_ind_exh(dmatrix, ind):
    n = ind.shape[0]
    change = True
    while change:
        bestInd = ind.copy()
        ind = ind.copy()
        bestDiff = 0
        iters = 3000
        change = False
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
                change = True
            #Perm2 102
            ind[i] = jval
            ind[j] = ival
            ind[k] = kval
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()
                change = True
            #Perm3 120
            ind[i] = jval
            ind[j] = kval
            ind[k] = ival
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()
                change = True
            #Perm4 201
            ind[i] = kval
            ind[j] = ival
            ind[k] = jval    
            fit = fitness(ind, dmatrix)
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()
                change = True
            #Perm5 210
            ind[i] = kval
            ind[j] = jval
            ind[k] = ival
            diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
            if diff > bestDiff:
                bestDiff = diff
                bestInd = ind.copy()
                change = True
            #Reset individual
            ind[i] = ival
            ind[j] = jval
            ind[k] = kval
            if bestDiff > 0:
                break
        ind = bestInd
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



@nb.njit()
def fast_swap3_lso_pop_exh(dmatrix, pop):
    n = pop.shape[1]
    for id in range(pop.shape[0]):
        #print(id, "pop[k]b", fitness(pop[id], dmatrix))
        change = True
        while change:
            bestInd = pop[id].copy()
            ind = pop[id].copy()
            bestDiff = 0
            iters = 500
            change = False
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
                    change = True
                #Perm2 102
                ind[i] = jval
                ind[j] = ival
                ind[k] = kval
                diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
                if diff > bestDiff:
                    bestDiff = diff
                    bestInd = ind.copy()
                    change = True
                #Perm3 120
                ind[i] = jval
                ind[j] = kval
                ind[k] = ival
                diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
                if diff > bestDiff:
                    bestDiff = diff
                    bestInd = ind.copy()
                    change = True
                #Perm4 201
                ind[i] = kval
                ind[j] = ival
                ind[k] = jval    
                fit = fitness(ind, dmatrix)
                diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
                if diff > bestDiff:
                    bestDiff = diff
                    bestInd = ind.copy()
                    change = True
                #Perm5 210
                ind[i] = kval
                ind[j] = jval
                ind[k] = ival
                diff = initialTotal - computeTotal(dmatrix, ind, i, slicei, j, slicej, sliceklow, k)
                if diff > bestDiff:
                    bestDiff = diff
                    bestInd = ind.copy()
                    change = True
                #Reset individual
                ind[i] = ival
                ind[j] = jval
                ind[k] = kval
                if bestDiff > 0:
                    break
            pop[id] = bestInd
            #print(id, "pop[k]a", fitness(pop[id], dmatrix))
        #print(id, "pop[k]final", fitness(pop[id], dmatrix))


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
def fast_opt2_lso(dmatrix, pop):
    n = pop.shape[1]
    intervalSize = 50
    for k in range(pop.shape[0]):
        bestDiff = 0
        besti = -1
        bestj = -1
        #iters = 500
        inv = False
        intervalStart = np.random.randint(0,n-(intervalSize-1))

        for i in range(intervalStart, intervalStart+intervalSize-1):
            for j in range(i+2, intervalStart+intervalSize):
                total = compute_opt_total(dmatrix, pop[k], i, j, n)
                pop[k][i:j] = np.flip(pop[k][i:j])
                total -= compute_opt_total(dmatrix, pop[k], i, j, n)
                if total > bestDiff:
                    bestDiff = total
                    besti = i
                    bestj = j
                    inv = False
                #Undo operation
                pop[k][i:j] = np.flip(pop[k][i:j])
                checki = max(0, intervalStart-1)
                #check inverse flip
                total = fitness2(pop[k, checki:intervalStart+intervalSize+2], dmatrix)
                total += dmatrix[pop[k, n-1], pop[k,0]]
                pop[k][intervalStart:i] = np.flip(pop[k][intervalStart:i])
                pop[k][j:intervalStart+intervalSize] = np.flip(pop[k][j:intervalStart+intervalSize])
                total -= fitness2(pop[k, checki:intervalStart+intervalSize+2], dmatrix)
                total -= dmatrix[pop[k,n-1], pop[k,0]]
                if total > bestDiff:
                    bestDiff = total
                    besti = i
                    bestj = j
                    inv = True
                #Undo inverse
                pop[k][j:intervalStart+intervalSize] = np.flip(pop[k][j:intervalStart+intervalSize])
                pop[k][intervalStart:i] = np.flip(pop[k][intervalStart:i])
        if inv:
            pop[k][intervalStart:besti] = np.flip(pop[k][intervalStart:besti])
            pop[k][bestj:intervalStart+intervalSize] = np.flip(pop[k][bestj:intervalStart+intervalSize])
        else:
            pop[k][besti:bestj] = np.flip(pop[k][besti:bestj])


@nb.njit()
def fast_opt2_lso_exh(dmatrix, pop):
    n = pop.shape[1]
    for k in range(pop.shape[0]):
        change = True
        while change:
            change = False
            bestDiff = 0
            besti = -1
            bestj = -1
            iters = 1000
            inv = False
            for a in range(iters):
                i = np.random.randint(0,n-1)
                j = np.random.randint(i+2,n+1)
                total = compute_opt_total(dmatrix, pop[k], i, j, n)
                pop[k][i:j] = np.flip(pop[k][i:j])
                total -= compute_opt_total(dmatrix, pop[k], i, j, n)

                if total > bestDiff:
                    bestDiff = total
                    besti = i
                    bestj = j
                    inv = False
                    change = True
                #Undo operation
                pop[k][i:j] = np.flip(pop[k][i:j])

                #check inverse flip
                total = compute_opt_inv_total(dmatrix, pop[k], i, j, n)
                pop[k][0:i] = np.flip(pop[k][0:i])
                pop[k][j:n] = np.flip(pop[k][j:n])
                total -= compute_opt_inv_total(dmatrix, pop[k], i, j, n)
                if total > bestDiff:
                    bestDiff = total
                    besti = i
                    bestj = j
                    inv = True
                    change = True
                #Undo inverse
                pop[k][j:n] = np.flip(pop[k][j:n])
                pop[k][0:i] = np.flip(pop[k][0:i])
            if inv:
                pop[k][0:besti] = np.flip(pop[k][0:besti])
                pop[k][bestj:n] = np.flip(pop[k][bestj:n])
            else:
                pop[k][besti:bestj] = np.flip(pop[k][besti:bestj])


@nb.njit()
def opt2_lso_pop(dmatrix, pop):
    n = pop.shape[1]
    for k in range(pop.shape[0]):
        bestDiff = 0
        besti = -1
        bestj = -1
        inv = False
        for i in range(n-1):
            for j in range(i+1, n+1):
                total = compute_opt_total(dmatrix, pop[k], i, j, n)
                pop[k][i:j] = np.flip(pop[k][i:j])
                total -= compute_opt_total(dmatrix, pop[k], i, j, n)

                if total > bestDiff:
                    bestDiff = total
                    besti = i
                    bestj = j
                    inv = False
                #Undo operation
                pop[k][i:j] = np.flip(pop[k][i:j])

                #check inverse flip
                total = compute_opt_inv_total(dmatrix, pop[k], i, j, n)
                pop[k][0:i] = np.flip(pop[k][0:i])
                pop[k][j:n] = np.flip(pop[k][j:n])
                total -= compute_opt_inv_total(dmatrix, pop[k], i, j, n)
                if total > bestDiff:
                    bestDiff = total
                    besti = i
                    bestj = j
                    inv = True
                #Undo inverse
                pop[k][j:n] = np.flip(pop[k][j:n])
                pop[k][0:i] = np.flip(pop[k][0:i])
        if inv:
            pop[k][0:besti] = np.flip(pop[k][0:besti])
            pop[k][bestj:n] = np.flip(pop[k][bestj:n])
        else:
            pop[k][besti:bestj] = np.flip(pop[k][besti:bestj])


@nb.njit()
def opt2_lso_pop_exh(dmatrix, pop):
    n = pop.shape[1]
    for k in range(pop.shape[0]):
        change = True
        while change:
            bestDiff = 0
            besti = -1
            bestj = -1
            inv = False
            change = False
            for i in range(n-1):
                for j in range(i+1, n+1):
                    total = compute_opt_total(dmatrix, pop[k], i, j, n)
                    pop[k][i:j] = np.flip(pop[k][i:j])
                    total -= compute_opt_total(dmatrix, pop[k], i, j, n)

                    if total > bestDiff:
                        bestDiff = total
                        besti = i
                        bestj = j
                        inv = False
                        change = True
                        
                    #Undo operation
                    pop[k][i:j] = np.flip(pop[k][i:j])

                    #check inverse flip
                    total = compute_opt_inv_total(dmatrix, pop[k], i, j, n)
                    pop[k][0:i] = np.flip(pop[k][0:i])
                    pop[k][j:n] = np.flip(pop[k][j:n])
                    total -= compute_opt_inv_total(dmatrix, pop[k], i, j, n)
                    if total > bestDiff:
                        bestDiff = total
                        besti = i
                        bestj = j
                        inv = True
                        change = True
                    #Undo inverse
                    pop[k][j:n] = np.flip(pop[k][j:n])
                    pop[k][0:i] = np.flip(pop[k][0:i])
            if inv:
                pop[k][0:besti] = np.flip(pop[k][0:besti])
                pop[k][bestj:n] = np.flip(pop[k][bestj:n])
            else:
                pop[k][besti:bestj] = np.flip(pop[k][besti:bestj])
            #print(k, "pop[k]", pop[k])

@nb.njit()
def compute_opt_total(dmatrix, curr_route, i,j, n):
    slicei = max(0, i-1)
    total = fitness2(curr_route[slicei:j+1], dmatrix)
    total += dmatrix[curr_route[n-1], curr_route[0]]
    return total
@nb.njit()
def compute_opt_inv_total(dmatrix, curr_route, i,j, n):
    total = fitness2(curr_route[0:i+1], dmatrix)
    total += fitness2(curr_route[j-1:n],dmatrix)
    total += dmatrix[curr_route[n-1], curr_route[0]]
    return total

@nb.njit()
def fast_opt3_lso_ind(dmatrix, ind):
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
def fast_opt3_lso_ind_exh(dmatrix, ind):
    n = ind.shape[0]
    change = True
    bestInd = ind.copy()
    while change:
        change = False
        #i = 0, j = 1, k = 2
        bestFit = fitness(ind, dmatrix)
        #Maintain a gap of 2 between indices, as flip does not do anything otherwise
        iters = 4000
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
                change = True
            ind[i:k] = np.flip(ind[i:k])

            #Perm2 j,k
            ind[j:k] = np.flip(ind[j:k])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
                change = True
            ind[j:k] = np.flip(ind[j:k])

            #Perm3 i,j
            ind[i:j] = np.flip(ind[i:j])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
                change = True
            ind[i:j] = np.flip(ind[i:j])

            #Perm4 ij, jk
            ind[i:j] = np.flip(ind[i:j])
            ind[j:k] = np.flip(ind[j:k])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
                change = True
            ind[j:k] = np.flip(ind[j:k])
            ind[i:j] = np.flip(ind[i:j])

            #Perm5 ik, jk
            ind[i:k] = np.flip(ind[i:k])
            ind[j:k] = np.flip(ind[j:k])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
                change = True
            ind[j:k] = np.flip(ind[j:k])
            ind[i:k] = np.flip(ind[i:k])
            #Perm6 ik, ij
            ind[i:k] = np.flip(ind[i:k])
            ind[i:j] = np.flip(ind[i:j])
            fit = fitness(ind, dmatrix)
            if fit < bestFit:
                bestFit = fit
                bestInd = ind.copy()
                change = True
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
                change = True
            ind[j:k] = np.flip(ind[j:k])
            ind[i:j] = np.flip(ind[i:j])
            ind[i:k] = np.flip(ind[i:k])
        ind = bestInd
    return bestInd



@nb.njit()
def full_opt3_lso_ind_exh(dmatrix, ind):
    n = ind.shape[0]
    change = True
    bestInd = ind.copy()
    while change:
        change = False
        #i = 0, j = 1, k = 2
        bestFit = fitness(ind, dmatrix)
        #Maintain a gap of 2 between indices, as flip does not do anything otherwise
        for i in range(n-3):
            for j in range(i+2, n-1):
                for k in range(j+2, n+1):
                    #Perm1 i,k
                    ind[i:k] = np.flip(ind[i:k])
                    fit = fitness(ind, dmatrix)
                    if fit < bestFit:
                        bestFit = fit
                        bestInd = ind.copy()
                        change = True
                    ind[i:k] = np.flip(ind[i:k])

                    #Perm2 j,k
                    ind[j:k] = np.flip(ind[j:k])
                    fit = fitness(ind, dmatrix)
                    if fit < bestFit:
                        bestFit = fit
                        bestInd = ind.copy()
                        change = True
                    ind[j:k] = np.flip(ind[j:k])

                    #Perm3 i,j
                    ind[i:j] = np.flip(ind[i:j])
                    fit = fitness(ind, dmatrix)
                    if fit < bestFit:
                        bestFit = fit
                        bestInd = ind.copy()
                        change = True
                    ind[i:j] = np.flip(ind[i:j])

                    #Perm4 ij, jk
                    ind[i:j] = np.flip(ind[i:j])
                    ind[j:k] = np.flip(ind[j:k])
                    fit = fitness(ind, dmatrix)
                    if fit < bestFit:
                        bestFit = fit
                        bestInd = ind.copy()
                        change = True
                    ind[j:k] = np.flip(ind[j:k])
                    ind[i:j] = np.flip(ind[i:j])

                    #Perm5 ik, jk
                    ind[i:k] = np.flip(ind[i:k])
                    ind[j:k] = np.flip(ind[j:k])
                    fit = fitness(ind, dmatrix)
                    if fit < bestFit:
                        bestFit = fit
                        bestInd = ind.copy()
                        change = True
                    ind[j:k] = np.flip(ind[j:k])
                    ind[i:k] = np.flip(ind[i:k])
                    #Perm6 ik, ij
                    ind[i:k] = np.flip(ind[i:k])
                    ind[i:j] = np.flip(ind[i:j])
                    fit = fitness(ind, dmatrix)
                    if fit < bestFit:
                        bestFit = fit
                        bestInd = ind.copy()
                        change = True
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
                        change = True
                    ind[j:k] = np.flip(ind[j:k])
                    ind[i:j] = np.flip(ind[i:j])
                    ind[i:k] = np.flip(ind[i:k])
        ind = bestInd
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
        iters = 300
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
        pop[id] = bestInd

@nb.njit()
def faster_opt3_lso_exh(dmatrix, pop):
    n = pop.shape[1]
    for id in range(pop.shape[0]):
        bestInd = pop[id].copy()
        change = True
        while change:
            change = False
            ind = pop[id].copy()
            #i = 0, j = 1, k = 2
            bestFit = fitness(ind, dmatrix)
            origFit = bestFit
            #Maintain a gap of 2 between indices, as flip does not do anything otherwise
            iters = 500
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
                    change = True
                ind[i:k] = np.flip(ind[i:k])

                #Perm2 j,k
                ind[j:k] = np.flip(ind[j:k])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()
                    change = True
                ind[j:k] = np.flip(ind[j:k])

                #Perm3 i,j
                ind[i:j] = np.flip(ind[i:j])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()
                    change = True
                ind[i:j] = np.flip(ind[i:j])

                #Perm4 ij, jk
                ind[i:j] = np.flip(ind[i:j])
                ind[j:k] = np.flip(ind[j:k])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()
                    change = True
                ind[j:k] = np.flip(ind[j:k])
                ind[i:j] = np.flip(ind[i:j])

                #Perm5 ik, jk
                ind[i:k] = np.flip(ind[i:k])
                ind[j:k] = np.flip(ind[j:k])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()
                    change = True
                ind[j:k] = np.flip(ind[j:k])
                ind[i:k] = np.flip(ind[i:k])
                #Perm6 ik, ij
                ind[i:k] = np.flip(ind[i:k])
                ind[i:j] = np.flip(ind[i:j])
                fit = fitness(ind, dmatrix)
                if fit < bestFit:
                    bestFit = fit
                    bestInd = ind.copy()
                    change = True
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
                    change = True
                ind[j:k] = np.flip(ind[j:k])
                ind[i:j] = np.flip(ind[i:j])
                ind[i:k] = np.flip(ind[i:k])

                if bestFit < origFit:
                    break
            pop[id] = bestInd
            #print(id, "popi", pop[id], fitness(pop[id], dmatrix))

#def eax(dmatrix, pop):
@nb.njit()
def orOptAll(dmatrix, pop):
    for i in range(pop.shape[0]):
        pop[i] = orOptMove(dmatrix, pop[i])



@nb.njit()
def orOptMove(dmatrix, ind):
    n = ind.shape[0]
    bestSolution = ind.copy()
    bestGain = 0
    for seg in range(1, 4):
        for i in range(n):
            j = (i + seg) % n
            x1 = ind[i]
            x2 = ind[(i+1)%n]
            y1 = ind[j]
            y2 = ind[(j+1)%n]

            for startPos in range(seg+1, n):
                k = (i+startPos) % n
                z1 = ind[k]
                z2 = ind[(k+1)%n]
                gain = gainFromOpt(x1, x2, y1, y2, z1, z2, dmatrix)
                if gain > bestGain:
                    bestGain = gain
                    bestSolution = shift(ind, i,j,k)
    return bestSolution                

@nb.njit()
def shift(ind, i, j, k):
    n = ind.shape[0]
    newRoute = np.zeros(n, dtype=np.int64)
    seglength = (j-i+n) % n
    shiftLength = (k-i+n) % n
    segment = np.zeros(seglength, dtype=np.int64)
    for idx in range(0, seglength):
        segment[idx] = ind[(idx+i+1)%n]
    newRoute[k] = ind[k]
    insertPos = (i + 1) % n
    for i in range(seglength):
        newRoute[(insertPos + shiftLength)%n] = segment[i]
        insertPos = (insertPos + 1) % n
    left = n - seglength -1
    segset = set(segment)
    segset.add(ind[k])
    i = 0
    while left > 0:
        candidate = ind[(k+i)%n]
        if candidate not in segset:
            newRoute[(insertPos + shiftLength)%n] = candidate
            insertPos = (insertPos + 1) % n
            left -= 1
        i += 1
    return newRoute

#https://tsp-basics.blogspot.com/2017/03/or-opt.html
@nb.njit()
def gainFromOpt(x1, x2, y1, y2, z1, z2, dmatrix):
    # 1,2,3,4,x1,x2,a,b,c,y1,y2,5,6,7,z1,z2,-,-,...
    # 0,1,2,3,4 ,5 ,6,7,8,9 ,10,1,2,3,14,15
    # 1,2,3,4,x1,y2,-,-,-,z1,x2,a,b,c,y1,z2,-,-,...
    #x2 to y1 shifted to inbetween z1 and z2
    total = dmatrix[x1,x2] + dmatrix[y1,y2] + dmatrix[z1,z2]
    total -= dmatrix[z1,x2] + dmatrix[y1, z2] + dmatrix[x1,y2]
    return total

######################## Mutation Rate Related Stuff #########################
def combineAlphas(a1, a2):
    b = 2 * np.random.uniform() - 0.5 #Between -0.5 and 1.5
    a = a1 + b * (a2-a1) 
    return np.abs(a)

######################## MISC HELPER FUNCTIONS ############################

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
prog = R0975929()
prog.optimize("tour1000.csv")

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
