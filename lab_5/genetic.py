import random
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from IPython.display import clear_output

class Genetic_optimization:
    def __init__(self, Backpack, pop_num_max, top_best, prob_mutation, pop_size):
        self.Backpack = Backpack
        self.pop_num_max = pop_num_max
        self.top_best = top_best
        self.prob_mutation = prob_mutation
        self.pop_size = pop_size
        self.pop = self.create_first_population()
        self.best_combination = self.pop[0]
        self.best_weight, self.best_value = self.Backpack.find_sum(self.best_combination)
        self.run_already = False
        
    def create_first_population(self):
        pop = []
        for _ in range(self.pop_size):
            tmp = []
            not_added = [i for i in range(self.Backpack.items)]
            np.random.shuffle(not_added)
            while self.Backpack.find_sum(tmp) != -1 and len(not_added) >= 1:
                tmp.append(not_added[-1])
                not_added.pop()
            tmp.pop()
            pop.append(tmp)
        return pop

    def in_boundry(self, combination):
        if self.Backpack.find_sum(combination) == -1:
            while self.Backpack.find_sum(combination) == -1:
                combination = self.Backpack.sort_by_value(combination)
                combination.pop()
        return combination
        
    def crossover(self, p1, p2):
        divide_by = random.randrange(1, min(len(p1), len(p2)))
        child1, child2 = p1[:divide_by] + p2[divide_by:], p2[:divide_by] + p1[divide_by:]
        child1, child2 = list(set(child1)), list(set(child2))
        return child1, child2
        
    def mutation(self, gene):
        i, j = np.random.choice(len(gene), 2, replace=False)
        gene[i], gene[j] = gene[j], gene[i]
        return gene
        
    def birth_children(self):
        new_population = []
        fitnesses = {}
        for i in range(len(self.pop)):
            fitnesses[self.Backpack.find_sum(self.pop[i])[1]] = self.pop[i]
        sorted_fitnesses = dict(reversed(sorted(fitnesses.items())))
        parents = list(sorted_fitnesses.values())[0:self.top_best]
        weight_best_parent, price_best_parent = self.Backpack.find_sum(parents[0])
        if price_best_parent > self.best_value:
            self.best_combination = parents[0]
            self.best_weight = weight_best_parent
            self.best_value = price_best_parent
        for i in range(len(parents)):
            new_population.append(parents[i])
        pairs = [[parents[i], parents[j]] for i in range(len(parents)) for j in range(i + 1, len(parents))]
        for i in range(len(pairs)):
            child1, child2 = self.crossover(pairs[i][0], pairs[i][1])
            child1 = self.in_boundry(child1)
            child2 = self.in_boundry(child2)
            chance = random.uniform(0, 1)
            child = random.randrange(0, 1)
            if chance <= self.prob_mutation:
                if child == 0:
                    child1 = self.mutation(child1)
                else:
                    child2 = self.mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        self.pop = new_population
        weights = np.zeros(self.pop_size-1)
        values = np.zeros(self.pop_size-1)
        for i in range(self.pop_size-1):
            weights[i], values[i] = self.Backpack.find_sum(self.pop[i])
        return weights, values

    def run(self):
        pop_num = 1
        while pop_num != self.pop_num_max:
            self.birth_children()
            pop_num += 1
        return self.best_combination, self.best_weight, self.best_value
        
    def plot_dynamic(self, iter):
        if self.run_already:
            self.pop = self.create_first_population()
            self.best_combination = self.pop[0]
            self.best_weight, self.best_value = self.Backpack.find_sum(self.best_combination)
        pop_num = 1
        history = []
        while pop_num != self.pop_num_max:
            pop_w, pop_v = self.birth_children()
            history.append(pop_v)
            pop_num += 1
            if pop_num % iter:
                clear_output(wait=True)
                fig, axs = plt.subplots(3, sharex=True, sharey=True)
                axs[0].bar(list(range(self.pop_size-1)), pop_v, color = 'cornflowerblue', label = 'value')
                axs[0].set_title('Values, best: {i}'.format(i = self.best_value))
                axs[1].bar(list(range(self.pop_size-1)), pop_w, color = 'mediumseagreen', label = 'weight')
                axs[1].set_title('Weights, weight of best: {j}'.format(j = self.best_weight))
                history = np.array(history).T
                for i in range(len(history)):
                    axs[2].plot(list(range(len(history[i]))), history[i])
                history = history.T.tolist()
                plt.show()
        self.run_already = True
        print("Max weight allowed: {i}, the weight of found: {j}".format(i = self.Backpack.max_weight, j = self.best_weight))
        print("Max value found: {i}, sum of all values: {j}".format(i = self.best_value, j = self.Backpack.sum_of_all))
        
    def plot_best(self):
        if self.run_already:
            self.pop = self.create_first_population()
            self.best_combination = self.pop[0]
            self.best_weight, self.best_value = self.Backpack.find_sum(self.best_combination)
        pop_num = 1
        history = []
        while pop_num != self.pop_num_max:
            self.birth_children()
            history.append(self.best_value)
            pop_num += 1
        plt.plot([i for i in range(len(history))], history)
        plt.title("Best solution progression")
        plt.show()
        self.run_already = True
        print("Max weight allowed: {i}, the weight of found: {j}".format(i = self.Backpack.max_weight, j = self.best_weight))
        print("Max value found: {i}, sum of all values: {j}".format(i = self.best_value, j = self.Backpack.sum_of_all))