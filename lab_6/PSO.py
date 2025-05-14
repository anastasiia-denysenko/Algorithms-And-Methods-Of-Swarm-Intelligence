import random
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from copy import deepcopy
import matplotlib.pyplot as plt

class PSO:
    def __init__(self, func, num_iter, a1, a2, size_pop, v_min, v_max, training, testing, lim1, lim2):
        if not (0 <= a1 <= 4 and 0 <= a2 <= 4):
            raise ValueError(f"Accelerations must be in range [0, 4], got {a1}, {a2}")
        if v_max <= 0:
            raise ValueError(f"Maximum velocity must be > 0, got {v_max}")
        
        self.func = func
        self.num_iter = num_iter
        self.a1 = a1
        self.a2 = a2
        self.size_pop = size_pop
        self.v_min = v_min
        self.v_max = v_max
        self.training = training
        self.testing = testing
        self.lim1 = lim1
        self.lim2 = lim2
        self.curr_iter = 0
        self.w = 0.9
        
        self.pop = np.array([
            [np.random.uniform(lim1[0], lim1[1]),
             np.random.uniform(lim2[0], lim2[1])]
            for _ in range(size_pop)
        ])
        self.vel_b1 = np.random.uniform(v_min, v_max, size_pop)
        self.vel_b2 = np.random.uniform(v_min, v_max, size_pop)

        self.bests = deepcopy(self.pop)
        self.fitnesses = np.array([self.fitness(p) for p in self.pop])
        self.best_fitnesses = deepcopy(self.fitnesses)
        
        self.ideal = self.pop[0]
        self.ideal_fitness = self.fitnesses[0]
        self.history = []

    def put_in_bounds(self, vec):
        vec[0] = np.clip(vec[0], self.lim1[0], self.lim1[1])
        vec[1] = np.clip(vec[1], self.lim2[0], self.lim2[1])
        return vec

    def fitness(self, vec):
        return np.mean([abs(self.func([x, vec[0], vec[1]]) - y) for x, y in self.training])

    def update_best(self):
        for i in range(self.size_pop):
            fit = self.fitness(self.pop[i])
            if fit < self.best_fitnesses[i]:
                self.bests[i] = self.pop[i].copy()
                self.best_fitnesses[i] = fit
            if fit < self.ideal_fitness:
                self.ideal = self.pop[i].copy()
                self.ideal_fitness = fit
                self.history.append(self.ideal_fitness)

    def create_velocity_vector(self):
        r1 = np.random.rand()
        r2 = np.random.rand()
        for i in range(self.size_pop):
            cognitive = self.a1 * r1 * (self.bests[i] - self.pop[i])
            social = self.a2 * r2 * (self.ideal - self.pop[i])
            velocity = self.w * np.array([self.vel_b1[i], self.vel_b2[i]]) + cognitive + social

            velocity = np.clip(velocity, self.v_min, self.v_max)

            self.vel_b1[i], self.vel_b2[i] = velocity[0], velocity[1]

    def update_positions(self):
        for i in range(self.size_pop):
            self.pop[i][0] += self.vel_b1[i]
            self.pop[i][1] += self.vel_b2[i]
            self.pop[i] = self.put_in_bounds(self.pop[i])
        return self.pop

    def iter(self):
        self.update_best()
        self.create_velocity_vector()
        self.update_positions()
        self.curr_iter += 1
        self.w = max(0.4, self.w * 0.99) 

    def run(self):
        while self.curr_iter < self.num_iter:
            self.iter()
        return self.ideal, self.ideal_fitness
        
    def plot(self):
        self.run()
        plt.scatter([i[0] for i in self.training], [i[1] for i in self.training], label = 'Training')
        plt.scatter([i[0] for i in self.testing], [i[1] for i in self.testing], label = 'Testing')
        minimum = min(min([i[0] for i in self.training]), min([i[0] for i in self.testing]))
        maximum = max(max([i[0] for i in self.training]), max([i[0] for i in self.testing]))
        x = np.linspace(minimum, maximum, 300)
        plt.plot(x, [self.func([i, self.ideal[0], self.ideal[1]]) for i in x], label = 'Prediction', linestyle = '--', c = 'green')
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.plot([i for i in range(len(self.history))], self.history)
        plt.show()