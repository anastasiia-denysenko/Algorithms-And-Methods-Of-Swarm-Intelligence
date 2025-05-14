import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import random
from functions import *  
from copy import deepcopy

class Cuculus_optimization:
    def __init__(self, func, p_detect, delta, iter, size_pop, limitations, minimize = True):
        if delta >= 1 or delta <= 0:
            raise ValueError("The value of delta has to be in (0, 1), but given value is: {delta}".format(delta = delta))
        self.func = func
        self.p_detect = p_detect
        self.delta = delta
        self.iter = iter
        self.size_pop = size_pop
        self.limitations = limitations
        self.minimize = minimize
        self.M = len(self.limitations)
        self.x_ideals = [self.limitations[i][0] for i in range(self.M)]
        self.pop = self.create_first_population()

    def create_first_population(self):
        pop = []
        for i in range(self.M):
            lim = self.limitations[i]
            xs = np.arange(lim[0], lim[1], abs(lim[0] - lim[1])/self.size_pop**2)
            xs = xs.reshape((self.size_pop, self.size_pop))
            pop.append(xs)
        return pop
    
    def choose_nest(self):
        return int(np.around((self.size_pop - 1)*random.uniform(0, 1), decimals=0))
    
    def put_in_boudry(self, i, j, k):
        min = self.limitations[i][0]
        max = self.limitations[i][1]
        diff = abs(min - max)
        if self.pop[i][j][k] >= max or self.pop[i][j][k] <= min:
            self.pop[i][j][k] = ((self.pop[i][j][k] - min) % diff + diff) % diff + min

    def update_positions(self):
        for i in range(self.M):
            min = self.limitations[i][0]
            max = self.limitations[i][1]
            diff = abs(min - max)
            ideals = deepcopy(self.x_ideals)
            for j in range(self.size_pop):
                for k in range(self.size_pop):
                    self.pop[i][j][k] = self.pop[i][j][self.choose_nest()] + self.delta * diff * (2*random.uniform(0, 1) - 1)
                    self.put_in_boudry(i, j, k)
                    all_dims = [i[j] for i in self.pop]
                    if np.mean(self.func(all_dims)) <= np.mean(self.func(self.x_ideals)) and self.minimize:
                        ideals[i] = self.pop[i][j][k]
                    elif np.mean(self.func(all_dims)) >= np.mean(self.func(self.x_ideals)) and not self.minimize:
                        ideals[i] = self.pop[i][j][k]
            self.x_ideals = ideals
        return self.x_ideals

    def find_worst(self, i):
        others = [self.pop[k][0] for k in range(0, self.M)]
        if self.minimize:
            worst = self.func(others)
            worst_index = 0
            for j in range(1, self.size_pop):
                others[i] = self.pop[i][j]
                other_func = self.func(others)
                if np.mean(other_func) > np.mean(worst):
                    worst = self.pop[i][j]
                    worst_index = j
        
        else:
            worst = self.func(others)
            worst_index = 0
            for j in range(1, self.size_pop):
                others[i] = self.pop[i][j]
                other_func = self.func(others)
                if np.mean(other_func) < np.mean(worst):
                    worst = self.pop[i][j]
                    worst_index = j
        return worst_index
            
    def modify_worst(self):
        if random.uniform(0, 1) < self.p_detect:
            return 
        else:
            for i in range(self.M):
                min = self.limitations[i][0]
                max = self.limitations[i][1]
                diff = abs(min - max)
                self.pop[i][self.find_worst(i)] += self.delta * diff * (2*random.uniform(0, 1) - 1)
                for k in self.pop[i][self.find_worst(i)]:
                    if k > max or k < min:
                        k = ((k - min) % diff + diff) % diff + min

    def run(self):
        all_ideals = []
        for _ in range(self.iter):
            self.x_ideals = self.update_positions()
            all_ideals.append(self.x_ideals)
            self.modify_worst()
        return all_ideals
    
    def plot(self):
        all_ideals = self.run()
        plt.plot([j for j in range(self.iter)], [self.func(np.array(row)) for row in all_ideals])
        plt.grid()
        plt.show()