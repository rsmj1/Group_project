import numpy as np
import random
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



class KnapSack:
  #The knapsack problem consist of a sack of elements, each with a weight and a value.
  #We have a weight capacity, and under this, we want to maximize the values of the chosen items.
  def __init__(self, numObjects = 10):
    self.values = 2**np.random.randn(numObjects)
    self.weights = 2**np.random.randn(numObjects)
    self.capacity = 0.25 * np.sum(self.weights)


class Individual:
  #Individuals from the population for the algorithm
  #Each will be initialized randomly and represent a permutation of the objects in the knapsack
  def __init__(self, numObjects = None, order = None, prob = None):
    if order is None:
      self.order = np.random.permutation(numObjects)
    else:
      self.order = order
    #self adaptivity parameter
    if prob is None:
      self.prob = 0.05
    else:
      self.prob = prob



def fitness(knapsack, individual):
  value = 0
  remainingCapacity = knapsack.capacity
  #Go through the ordering imposed by the individual, and take the element on that spot and add to the total value until we reach max capacity
  for elem in individual.order:
    if knapsack.weights[elem] > remainingCapacity:
      break
    value += knapsack.values[elem]
    remainingCapacity -= knapsack.weights[elem]
  return value

def solution(knapsack, individual):
  objects = []
  remainingCapacity = knapsack.capacity
  #Go through the ordering imposed by the individual, and take the element on that spot and add to the total value until we reach max capacity
  for elem in individual.order:
    if knapsack.weights[elem] > remainingCapacity:
      break
    #objects += [(elem, knapsack.weights[elem], knapsack.values[elem])]
    objects += [elem]
    remainingCapacity -= knapsack.weights[elem]
  return np.array(objects)

#Mutate inplace, so returns nothing - INVERSION MUTATION
def mutation(individual):
  #Mutate with probability prob from the individual
  if np.random.uniform() < individual.prob:
    i = np.random.randint(0,len(individual.order))
    j = np.random.randint(0,len(individual.order))
    individual.order[i:j] = individual.order[i:j][::-1]
  

#Mutate inplace, so returns nothing - SWAP MUTATION
def mutation(individual):
  #Mutate with probability prob from the individual
  if np.random.uniform() < individual.prob:
    i = np.random.randint(0,len(individual.order))
    j = np.random.randint(0,len(individual.order))
    tmp = individual.order[i]
    individual.order[i] = individual.order[j]
    individual.order[j] = tmp    

#Note that since we are working with permutations, one has to be careful when combining, that we still get a proper permutation out
def recombination(sack, p1, p2):
  #Subset based recombination
  sol1 = solution(sack, p1)
  sol2 = solution(sack, p2)
  offspring = np.intersect1d(sol1, sol2)
  #print("Intersection:", offspring)
  #put offspring in a random order
  offspring = np.random.permutation(offspring)
  #print("After permutation", offspring)
  symdiff = np.setdiff1d(np.union1d(sol1, sol2), offspring)
  #print("symdiff:", symdiff)
  for elem in symdiff:
    if np.random.uniform() <= 0.5:
      offspring = np.append(offspring, elem)
  #print("offspring after adding:", offspring)
  elemsLeft = np.random.permutation(np.setdiff1d(np.array(range(0, len(p1.order))), offspring))
  #print("elemsleft:", elemsLeft)
  for elem in elemsLeft:
    offspring = np.append(offspring, elem)

  beta = 2.*np.random.uniform() - 0.5
  newAlpha = p1.prob + beta * (p2.prob - p1.prob)
  
  #print("final offspring:", offspring)
  #print("With alpha:", newAlpha)
  newInd = Individual(order = offspring, prob = newAlpha)
  return newInd

#K-tournament selection
def selection(sack, pop, k=5):
  selected = np.random.choice(pop, k)
  choice = np.argmax(np.array([fitness(sack, x) for x in selected]))
  return selected[choice]

#lambda + mu elimination
def elimination(sack, pop, offspring, l):
  combination = np.append(pop, offspring)
  pred = np.array([fitness(sack, x) for x in combination])
  ordering = np.argsort(pred)
  choices = combination[ordering][l:]

  return choices


class Parameters:
  def __init__(self, l, m, k, its):
    self.la = l
    self.mu = m
    self.k = k
    self.its = its



def knapsackAlgo(sack, p):
  numObjects = len(sack.values)

  #Initial population
  population = initPopulation(numObjects, p.la)

  #Main loop
  maxIter = p.its
  for i in range(maxIter):
    
    offspring = np.empty(p.mu, dtype = Individual)
    #Recombination (D' X D' -> D')
    for j in range(p.mu):
      p1 = selection(sack, population, p.k)
      p2 = selection(sack, population, p.k)
      offspring[j] = recombination(sack, p1, p2)
      mutation(offspring[j])

    #Mutation
    for elem in population:
      mutation(elem)

    #Elimination
    population = elimination(sack, population, offspring, p.la)

    #Evaluation
    objectiveValues = np.array([fitness(sack, ind) for ind in population])
    print("Iteration: ", i, ", Mean fitness:", np.mean(objectiveValues), " Max fitness:", np.max(objectiveValues))
    

  return None


def initPopulation(numObjects, popSize):
  return np.array([Individual(numObjects) for i in range(popSize)])

def printPopulation(pop, sack = None):
  print("Population with size ", len(pop))
  for i, p in enumerate(pop):
    print(i,": ",  p.order)
    if sack is not None:
      print("Fitness: ", fitness(sack, p), ", Solution: ", solution(sack, p))
      
      
      

#val = 2**np.random.randn(10)
#print(val)
size = 1000

sack = KnapSack(size)
params = Parameters(100, 100, 5, 25)

print("Problem instance:")
print("Capacity:", sack.capacity)
print("Values:", sack.values)
print("Weights", sack.weights)
print("-----------------------------------")

knapsackAlgo(sack, params)



#ind = Individual(size)
#print("Individual before:", ind.order)
#mutation(ind)
#print("Ind after:", ind.order)
#print("Fitness", fitness(sack, ind))

#inds = initPopulation(size, size)
#inds2 = initPopulation(size, size)
#print("Elimination choices:", elimination(sack, inds, inds2, size))
# for i in inds:
#   print("Individual:", i.order)
#   print("Fitness:", fitness(sack, i))
#   print("Objects:", solution(sack, i))


#print("Combining:", ind.order, ", with solution:", solution(sack, ind))
#print("+ Combining:", inds[0].order, ", with solution:", solution(sack, inds[0]))

#recombination(sack, ind, inds[0])


