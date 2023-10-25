import Reporter
import numpy as np
import random

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

        print("Distance Matrix: \n", distanceMatrix)
		#Parameters
        lam = params.la #Population size
        mu = params.mu  #Offspring size  
        k = params.k    #K-tournament selection param  
        iterations = params.iterations
        numCities = len(distanceMatrix[0])

    
        #Initialise the population
        population = initPopulation(numCities, lam)
        i = 0
        yourConvergenceTestsHere = True
        while( yourConvergenceTestsHere ):
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])

			# Your code here.
			#Create offspring
            offspring = np.empty(mu, dtype = Individual)

        # Select from the population:
			#Recombination (D' X D' -> D')
            for j in range(mu):
                p1 = selection(distanceMatrix, population, k)
                p2 = selection(distanceMatrix,population, k)
                #offspring[j] = generate_child_chromosome(p1, p2)
                offspring[j] = pmx(p1, p2)
                swapMutation(offspring[j])

			#Mutation
            for elem in population:
                swapMutation(elem)



			#Elimination
            population = elimination(distanceMatrix, population, offspring, lam)

      

			#Evaluation
            objectiveValues = np.array([fitness(distanceMatrix, ind) for ind in population])
            print("Iteration: ", i, ", Mean fitness:", np.mean(objectiveValues), " Min fitness:", np.min(objectiveValues))

            i += 1

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			#print("hello", timeLeft)
            if i >= iterations:
                break

            if timeLeft < 0:
                break
            
        print("DONE!!!")
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
      self.alpha = max(0.01, 0.1 + 0.02*np.random.normal())
    else:
      self.alpha = alpha

class Parameters:
  def __init__(self, lambd, mu, k, its):
    self.la = lambd
    self.mu = mu
    self.k = k
    self.iterations = its



def initPopulation(numCities, popSize):
  return np.array([Individual(numCities) for i in range(popSize)])


def fitness(dmatrix, individual):
	distance = 0
	n = len(dmatrix[0])
	route = individual.route
	#print("Route length:", len(route))
	for i in range(0,n-1):
		distance += dmatrix[route[i], route[i+1]]  
	distance += dmatrix[route[n-1], route[0]]      
	return distance

def cycleSolution(dmatrix, individual):
     return []





####################### MUTATIONS #############################

#Mutate inplace, so returns nothing - INVERSION MUTATION
def invMutation(individual):
  #Mutate with probability prob from the individual
  if np.random.uniform() < individual.alpha:
    i = np.random.randint(0,len(individual.route)-1)
    j = np.random.randint(i+1,len(individual.route))
    individual.route[i:j] = individual.route[i:j][::-1]

  

#Mutate inplace, so returns nothing - SWAP MUTATION
def swapMutation(individual):
  #Mutate with probability prob from the individual
  if np.random.uniform() < individual.alpha:
    i = np.random.randint(0,len(individual.route))
    j = np.random.randint(0,len(individual.route))
    tmp = individual.route[i]
    individual.route[i] = individual.route[j]
    individual.route[j] = tmp




def edge_crossover(parent1, parent2):

    parents = {'Parent 1': list(parent1.route),
                'Parent 2': list(parent2.route)}
    """
    Perform Edge Crossover to create a CHILD chromosome.

    Edge Crossover is based on the idea that offspring should be created as
    far as possible using only edges that are present in (one of) the parents.
    The algorithm ensures that common edges are preserved.

    Parameters:
    parents (dict): A dictionary containing parent nodes and their neighbors.
        Example: {'Parent 1': ['A', 'B', 'F', 'E', 'D', 'G', 'C'],
                  'Parent 2': ['G', 'F', 'A', 'B', 'C', 'D', 'E']}

    Returns:
    list: The generated CHILD chromosome, which is a list of nodes.

    Example:
    parents = {'Parent 1': ['A', 'B', 'F', 'E', 'D', 'G', 'C'],
               'Parent 2': ['G', 'F', 'A', 'B', 'C', 'D', 'E']}
    child_chromosome = edge_crossover(parents)
    print("CHILD Chromosome:", child_chromosome)
    """

    # Step 1: Construct the edge table
    edge_table = {}
    all_nodes = set(parents['Parent 1'] + parents['Parent 2'])
    for node in all_nodes:
        common_edges = set(parents['Parent 1']) & set(parents['Parent 2'])
        if node in parents['Parent 1']:
            common_edges.update(parents['Parent 1'])
        if node in parents['Parent 2']:
            common_edges.update(parents['Parent 2'])
        common_edges.remove(node)
        edge_table[node] = list(common_edges)

    # Step 2: Pick an initial element at random and put it in the offspring
    offspring = []
    initial_element = random.choice(parents[random.choice(list(parents.keys()))])
    offspring.append(initial_element)

    # Step 3: Set the variable current_element
    current_element = initial_element

    # Step 4: Remove all references to current_element from the table
    for node, common_edges in edge_table.items():
        if current_element in common_edges:
            common_edges.remove(current_element)

    # Steps 5 and 6: Continue building the offspring
    while len(offspring) < len(all_nodes):
        if not edge_table[current_element]:
            # Current element's edge list is empty
            # Examine the other end of the offspring for extension
            remaining_nodes = list(all_nodes - set(offspring))
            if remaining_nodes:
                next_element = random.choice(remaining_nodes)
            else:
                break  # No more nodes to add, exit the loop
        else:
            # Examine the list for the current element
            possible_next_elements = edge_table[current_element]

            # Find the next element with the shortest list
            shortest_list_length = min(map(lambda e: len(edge_table[e]), possible_next_elements))
            shortest_lists = [e for e in possible_next_elements if len(edge_table[e]) == shortest_list_length]

            # Randomly select the next element from the shortest lists
            next_element = random.choice(shortest_lists)

        # Update the current element and add it to the offspring
        current_element = next_element
        offspring.append(current_element)

    return Individual(route = offspring)

# edge crossover attempt 2
def generate_child_chromosome(parent1, parent2):

    parents = {'Parent 1': list(parent1.route),
                'Parent 2': list(parent2.route)}

    """
    Generate a CHILD chromosome using information from parent nodes.

    This function generates a CHILD chromosome based on a neighbor list
    and a set of parent nodes. It follows a specific algorithm to create
    the CHILD chromosome.

    Parameters:
    parents (dict): A dictionary containing parent nodes and their neighbors.
        Example: {'Parent 1': ['A', 'B', 'F', 'E', 'D', 'G', 'C'],
                  'Parent 2': ['G', 'F', 'A', 'B', 'C', 'D', 'E']}

    Returns:
    list: The generated CHILD chromosome, which is a list of nodes.

    Example:
    parents = {'Parent 1': ['A', 'B', 'F', 'E', 'D', 'G', 'C'],
               'Parent 2': ['G', 'F', 'A', 'B', 'C', 'D', 'E']}
    child_chromosome = generate_child_chromosome(parents)
    print("CHILD Chromosome:", child_chromosome)
    """

    # Step 1: Generate the neighbor list
    neighbor_list = {}
    all_nodes = set(parents['Parent 1'] + parents['Parent 2'])
    for node in all_nodes:
        neighbors = set(parents['Parent 1']) & set(parents['Parent 2'])
        if node in parents['Parent 1']:
            neighbors.update(parents['Parent 1'])
        if node in parents['Parent 2']:
            neighbors.update(parents['Parent 2'])
        neighbors.remove(node)
        neighbor_list[node] = list(neighbors)

    # Step 2: Initialize the CHILD chromosome
    CHILD = []

    # Step 3: Choose the first node from a random parent
    X = random.choice(parents[random.choice(list(parents.keys()))])

    # Step 4: Generate the CHILD chromosome
    while len(CHILD) < len(all_nodes):
        # Append X to CHILD
        CHILD.append(X)

        # Remove X from Neighbor Lists
        for node, neighbors in neighbor_list.items():
            if X in neighbors:
                neighbors.remove(X)

        if not neighbor_list[X]:
            # X's neighbor list is empty
            remaining_nodes = list(all_nodes - set(CHILD))
            if remaining_nodes:
                Z = random.choice(remaining_nodes)
            else:
                break  # No more nodes to add, exit the loop
        else:
            # Determine the neighbor of X with the fewest neighbors
            min_neighbors = min(neighbor_list[X], key=lambda node: len(neighbor_list[node]))
            neighbors_with_min = [node for node in neighbor_list[X] if len(neighbor_list[node]) == len(neighbor_list[min_neighbors])]
            Z = random.choice(neighbors_with_min)

        # Update X
        X = Z

    # Return the CHILD chromosome
    return Individual(route = CHILD, alpha = combineAlphas(parent1.alpha, parent2.alpha))

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

def recursive_fill(index,index_set,item,offspring,candidate):
    if index not in index_set:
        offspring[index] = item
        index_set.add(index)
    else:
        new_item = offspring[index]
        new_index = candidate.index(new_item)
        recursive_fill(new_index,index_set,item,offspring,candidate)

### END PMX



########################### Elimination and Selection ################################
#K-tournament selection
def selection(dmatrix, pop, k=5):
  selected = np.random.choice(pop, k)
  min = np.inf
  for ind in selected:
      if fitness(dmatrix, ind) < min:
        min = fitness(dmatrix, ind)

  choice = np.argmin(np.array([fitness(dmatrix, x) for x in selected]))
  
  assert fitness(dmatrix, selected[choice]) == min, "did not choose smallest value in selection"
  return selected[choice]

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


def combineAlphas(a1, a2):
    b = 2 * np.random.uniform() - 0.5
    a = a1 + b * (a2-a1)
    return a


#At some point we need to translate our path into a cycle
def solutionToCycle():
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


#printIndividual(testInd, testArray1)



prog = TspProg()
params = Parameters(lambd=200, mu=200, k=5, its=500)
prog.optimize("tour50.csv", params)