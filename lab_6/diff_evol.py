import numpy as np
import random
import matplotlib.pyplot as plt

class DifferentialEvolution:
    def __init__(self, func, num_iter, pop_size, lim1, lim2, training, testing, cross_prob, diff_weight):
        self.func = func
        self.num_iter = num_iter
        self.pop_size = pop_size
        self.lim1 = lim1
        self.lim2 = lim2
        self.training = training
        self.testing = testing
        self.cross_prob = cross_prob
        self.diff_weight = diff_weight

        self.population = np.array([
            [np.random.uniform(lim1[0], lim1[1]),
             np.random.uniform(lim2[0], lim2[1])]
            for _ in range(pop_size)
        ])

        self.best = self.population[0]
        self.best_score = self.fitness(self.best)
        self.history = []

    def fitness(self, b_vec):
        return sum(abs(self.func([x, b_vec[0], b_vec[1]]) - y) for x, y in self.training)

    def put_in_bounds(self, vec):
        vec[0] = np.clip(vec[0], self.lim1[0], self.lim1[1])
        vec[1] = np.clip(vec[1], self.lim2[0], self.lim2[1])
        return vec

    def mutate(self, a, b, c):
        return a + self.diff_weight * (b - c)

    def iterate(self):
        new_population = []
        for i in range(self.pop_size):
            a, b, c = self.population[np.random.choice(self.pop_size, 3, replace=False)]
            mutant = self.mutate(a, b, c)
            mutant = self.put_in_bounds(mutant)

            trial = np.copy(self.population[i])
            for j in range(2):
                if np.random.rand() < self.cross_prob:
                    trial[j] = mutant[j]

            if self.fitness(trial) < self.fitness(self.population[i]):
                new_population.append(trial)
                if self.fitness(trial) < self.best_score:
                    self.best = trial
                    self.best_score = self.fitness(trial)
                    self.history.append(self.best_score)
            else:
                new_population.append(self.population[i])
        self.population = np.array(new_population)

    def run(self):
        for _ in range(self.num_iter):
            self.iterate()
        return self.best

    def plot(self):
        self.run()
        plt.scatter([i[0] for i in self.training], [i[1] for i in self.training], label = 'Training')
        plt.scatter([i[0] for i in self.testing], [i[1] for i in self.testing], label = 'Testing')
        minimum = min(min([i[0] for i in self.training]), min([i[0] for i in self.testing]))
        maximum = max(max([i[0] for i in self.training]), max([i[0] for i in self.testing]))
        x = np.linspace(minimum, maximum, 300)
        plt.plot(x, [self.func([i, self.best[0], self.best[1]]) for i in x], label = 'Prediction', linestyle = '--', c = 'green')
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.plot([i for i in range(len(self.history))], self.history)
        plt.show()