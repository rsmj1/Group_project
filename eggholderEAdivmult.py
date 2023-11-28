import numpy as np
import random
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

""" A basic evolutionary algorithm. """
class eggholderEA:

	""" Initialize the evolutionary algorithm solver. """
	def __init__(self, fun):
		self.alpha = 0.05     		# Mutation probability
		self.lambdaa = 100     		# Population size
		self.mu = self.lambdaa * 2	# Offspring size
		self.k = 5            		# Tournament selection
		self.intMax = 500     		# Boundary of the domain, not intended to be changed.
		self.numIters = 30			# Maximum number of iterations
		self.origf = fun
		#self.objf = lambda x, pop=None, betaInit=0: self.sharedFitnessWrapper(fun, x, pop, betaInit) #I define it like this, such that the original code still runs
		self.objf = lambda x, pop=None: self.dominationFitnessWrapper(fun, x, pop)

	""" The main evolutionary algorithm loop. """
	def optimize( self, plotFun = lambda x : None):
		# Initialize population
		population = self.intMax * np.random.rand(self.lambdaa, 2)

		plotFun((population, self.intMax))
		for i in range(self.numIters):
			# The evolutionary algorithm
			start = time.time()
			selected = self.selection(population, self.k)
			offspring = self.crossover(selected)
			joinedPopulation = np.vstack((self.mutation(offspring, self.alpha), population))
			population = self.selection(joinedPopulation, self.k)
			itT = time.time() - start

			# Show progress
			fvals = self.origf(population)
			meanObj = np.mean(fvals, axis=0)
			bestObj = np.min(fvals, axis=0)
			print(f'{itT: .2f}s:\t Mean fitness = {meanObj[0]: .5f} \t Best fitness = {bestObj[0]: .5f}')
			plotFun((population, self.intMax))
		print('Done')

	""" Perform k-tournament selection to select pairs of parents. """
	def selection(self, population, k):
		selected = np.zeros((self.mu, 2))
		for ii in range( self.mu ):
			ri = random.choices(range(np.size(population,0)), k = self.k)
			min = np.argmin( self.objf(population[ri, :], population) )
			selected[ii,:] = population[ri[min],:]
		return selected

	def dominationFitnessWrapper(self, fun, x, pop):
		dominatedCounts = np.zeros( np.size(x,0) )
		xfvals = fun(x)
		fvals = fun(pop)
		for i, xi in enumerate(x):
			for y_fval in fvals:
				dominatedCounts[i] += int(np.all(y_fval <= xfvals[i,:]))


		
		return dominatedCounts

	def sharedFitnessWrapper(self, fun, x, pop = None, betaInit=0):
		if pop is None:
			return fun(x)
		
		alpha = 1
		sigma = self.intMax * 0.1 #This defines the max distance of individuals we will consider - decreasing this value, will make it closer to the original fitness function, as we are penalizing wiht less and less points
		modObjv = np.zeros(x.shape[0]) #Shape 0 is the amount of rows - we might have multiple individuals at once we compute the fitness for
		for i, xi in enumerate(x):
			ds = self.euclid_dists(xi, pop)
			beta = betaInit
			for d in ds:
				if d <= sigma:
					beta += 1 - (d/sigma)**alpha
			origVal = fun(xi)
			modObjv[i] = origVal*beta**np.sign(origVal)
			
		return modObjv

	
	
	#For fitness sharing we need some sort of distance function between two points. In this case, as it is a 2D problem, euclidean distance is an obvious choice.
	#We take the distance between the point and all other points in one function
	#Axis 1 is along the rows, which makes sense, assuming x is a row vector, and all points in Y are along the rows as well.
	def euclid_dists(self, x, Y):
		return np.linalg.norm(x-Y, axis=1)
	


	""" Perform box crossover as in the slides. """
	def crossover(self, selected):
		weights = 3*np.random.rand(self.lambdaa,2) - 1
		offspring = np.zeros((self.lambdaa, 2))
		lc = lambda x, y, w: np.clip(x + w * (y-x), 0, self.intMax)
		for ii, _ in enumerate(offspring):
			offspring[ii,:] = lc(selected[2*ii, :], selected[2*ii+1, :], weights[ii, :])
		return offspring

	""" Perform mutation, adding a random Gaussian perturbation. """
	def mutation(self, offspring, alpha):
		ii = np.where(np.random.rand(np.size(offspring,0)) <= alpha)
		offspring[ii,:] = offspring[ii,:] + 10*np.random.randn(np.size(ii),2)
		offspring[ii,0] = np.clip(offspring[ii,0], 0, self.intMax)
		offspring[ii,1] = np.clip(offspring[ii,1], 0, self.intMax)
		return offspring

	""" Eliminate the unfit candidate solutions. """
	def elimination(self, joinedPopulation, keep):
		fvals = self.objf(joinedPopulation)
		perm = np.argsort(fvals)
		survivors = joinedPopulation[perm[0:keep-1],:]
		return survivors

	"""Type of lambda + mu elimination based on """
	def shared_elimination(self, pop, keep):
		survivors = np.zeros((keep, 2))
		for i in range(keep):
			fvals = self.objf(pop, survivors[0:i-1,:], 1) #We need the 1 here, as we do not have the individual itself in the list of survivors
			idx = np.argmin(fvals)
			survivors[i] = pop[idx,:]

		return survivors

""" Compute the objective function at the vector of (x,y) values. """
def myfun(x):
	if np.size(x) == 2:
		x = np.reshape(x, (1,2))
	sas = np.sqrt(np.abs(x[:,0]+x[:,1]))
	sad = np.sqrt(np.abs(x[:,0]-x[:,1]))
	f = -x[:,1] * np.sin(sas) - x[:,0] * np.sin(sad)
	g = np.linalg.norm(x-np.array([250,250]), axis=1)
	return np.transpose(np.vstack((f,g))) #Returns multihead values

"""
Make a 3D visualization of the optimization landscape and the location of the
given candidate solutions (x,y) pairs in input[0].
"""
def plotPopulation3D(input):
	population = input[0]

	x = np.arange(0,input[1],0.5)
	y = np.arange(0,input[1],0.5)
	X, Y = np.meshgrid(x, y)
	Z = myfun(np.transpose(np.vstack((X.flatten(),Y.flatten()))))

	Z = np.reshape(Z[:,0], (np.size(x), np.size(y)))
	fig = plt.gcf()
	fig.clear()
	ax = fig.gca(projection='3d')
	ax.scatter(population[:,0], population[:,1], myfun(population)+1*np.ones(population.shape[0]), c='r', marker='o')
	ax.plot_surface(X, Y, np.reshape(Z, (np.size(x), np.size(y))), cmap=cm.coolwarm, linewidth=1, antialiased=False, alpha=0.2)
	plt.pause(0.05)

"""
Make a 2D visualization of the optimization landscape and the location of the
given candidate solutions (x,y) pairs in input[0].
"""
def plotPopulation2D(input):
	population = input[0]

	x = np.arange(0,input[1],1)
	y = np.arange(0,input[1],1)
	X, Y = np.meshgrid(x, y)
	Z = myfun(np.transpose(np.vstack((X.flatten(),Y.flatten()))))
	Z = np.reshape(Z[:,0], (np.size(x), np.size(y)))

	# Determine location of minimum
	rowMin = np.min(Z, axis=0)
	minj = np.argmin(rowMin)
	colMin = np.min(Z, axis=1)
	mini = np.argmin(colMin)

	fig = plt.gcf()
	fig.clear()
	ax = fig.gca()
	ax.imshow(Z)
	ax.scatter(population[:,0], population[:,1], marker='o', color='r')
	ax.scatter(mini, minj, marker='*', color='yellow')
	plt.pause(0.05)

""" Plots the Pareto front. """
def plotParetoFront(input):
	pop = input[0]
	fval = myfun(pop)
	fig = plt.gcf()
	fig.clear()
	ax = fig.gca()
	ax.scatter(fval[:,1], fval[:,0], marker='o', color='r')
	plt.pause(0.001)

eggEA = eggholderEA(myfun)
eggEA.optimize(plotPopulation2D)
plt.show()
