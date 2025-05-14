import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import random
from functions import *  
from copy import deepcopy

class Bat_optimization:
    def __init__(self, func, size_pop, iter, r, A, alpha, gamma, delta, f_min, f_max, limitations, minimize = True):
        if A > 2 or A < 1:
            raise ValueError("The value of A has to be in range [1, 2], but given value is {A}".format(A = A))
        if alpha <= 0 or alpha >= 1:
            raise ValueError("The value of alpha has to be in range (0, 1), but given value is {a}".format(a = alpha))
        if r < 0 or r > 1:
            raise ValueError("The value of r has to be in range [0, 1], but given value is {r}".format(r = r))
        if gamma <= 0 or gamma >= 1:
            raise ValueError("The value of gamma has to be in range (0, 1), but given value is {g}".format(g = gamma))
        if delta <= 0 or delta >= 1:
            raise ValueError("The value of delta has to be in range (0, 1), but given value is {d}".format(d = delta))
        if f_min < 0:
            raise ValueError("The value of f_min has to be greater than 0, but given value is {f}".format(f = f_min))
        if f_max <= f_min:
            raise ValueError("The value of f_max has to be greater than f_min, but given values are f_min: {f} and f_max {f1}".format(f = f_min, f1 = f_max))
        
        self.func = func
        self.size_pop = size_pop
        self.iter = iter
        self.r = r
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.f_min = f_min
        self.f_max = f_max
        self.limitations = limitations
        self.minimize = minimize
        self.M = len(self.limitations)
        self.pop = self.create_first_population()
        self.x_ideals = [self.limitations[i][0] for i in range(self.M)]
        self.velocity = [np.full((self.size_pop, self.size_pop), 0)]*self.M
        self.fk = self.f_min
        self.curr_iter = 0

    def create_first_population(self):
        pop = []
        for i in range(self.M):
            lim = self.limitations[i]
            xs = np.arange(lim[0], lim[1], abs(lim[0] - lim[1])/self.size_pop**2)
            xs = xs.reshape((self.size_pop, self.size_pop))
            pop.append(xs)
        return pop
    
    def put_in_boudry(self, i, j, k):
        min = self.limitations[i][0]
        max = self.limitations[i][1]
        diff = abs(min - max)
        if self.pop[i][j][k] >= max or self.pop[i][j][k] <= min:
            self.pop[i][j][k] = ((self.pop[i][j][k] - min) % diff + diff) % diff + min

    def update_f_v_x(self):
        self.fk = self.f_min + (self.f_max - self.f_min)*random.uniform(0, 1)
        for i in range(self.M):
            for j in range(self.size_pop):
                for k in range(self.size_pop):
                    self.velocity[i][j][k] += (self.x_ideals[i] - self.pop[i][j][k])*self.fk
                    self.pop[i][j][k] += self.velocity[i][j][k]
                    self.put_in_boudry(i, j, k)

    def x_curr(self):
        for i in range(self.M):
            min = self.limitations[i][0]
            max = self.limitations[i][1]
            diff = abs(min - max)
            ideals = deepcopy(self.x_ideals)
            for j in range(self.size_pop):
                for k in range(self.size_pop):
                    x_curr = self.pop[i][j][k] + self.delta * diff * (2*random.uniform(0, 1) - 1)
                    if x_curr >= max or x_curr <= min:
                        x_curr = ((x_curr - min) % diff + diff) % diff + min
                    tmp = np.array([[self.pop[l][j][k]] for l in range(self.M)])
                    tmp[i] = x_curr
                    if self.minimize and self.func(tmp) < self.func(np.array([self.pop[l][j][k] for l in range(self.M)])):
                        if random.uniform(0, 1) < self.A:
                            self.pop[i][j][k] = x_curr
                            self.A *= self.alpha
                            self.r *= (1 - np.e**(-self.gamma*self.curr_iter))
                        if self.func(tmp) < self.func(np.array([[i] for i in ideals])):
                            ideals[i] = x_curr
                    elif not self.minimize and self.func(tmp) > self.func(np.array([[self.pop[l][j][k]] for l in range(self.M)])):
                        if random.uniform(0, 1) < self.A:
                            self.pop[i][j][k] = x_curr
                            self.A *= self.alpha
                            self.r *= (1 - np.e**(-self.gamma*self.curr_iter))
                        if self.func(tmp) > self.func(np.array([[i] for i in ideals])):
                            ideals[i] = x_curr
            self.x_ideals = ideals

    def run(self):
        all_ideals = []
        for _ in range(self.iter):
            self.update_f_v_x()
            self.x_curr()
            all_ideals.append(self.x_ideals)
        return all_ideals
    
    def plot(self):
        all_ideals = self.run()
        plt.plot([j for j in range(self.iter)], [self.func(row) for row in all_ideals])
        plt.grid()
        plt.show()