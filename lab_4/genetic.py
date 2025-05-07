import random
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from IPython.display import clear_output

class Genetic_optimization:
    def __init__(self, Travelling_salesman, pop_num_max, top_best, prob_mutation, pop_size):
        self.Travelling_salesman = Travelling_salesman
        self.pop_num_max = pop_num_max
        self.top_best = top_best
        self.prob_mutation = prob_mutation
        self.pop_size = pop_size
        self.pop = self.create_first_population()
        self.best_route = self.pop[0]
        self.len_best_route = self.Travelling_salesman.route(self.Travelling_salesman.get_route(self.best_route))
        self.run_already = False
        
    def create_first_population(self):
        pop = np.array([np.random.permutation(self.Travelling_salesman.nums) for _ in range(self.pop_size)])
        return pop
        
    def crossover(self, parent1, parent2):
        a, b = sorted(np.random.choice(self.Travelling_salesman.nums, 2, replace=False))
        child = np.full((self.Travelling_salesman.nums, ), -1)
        child[a:b] = parent1[a:b]
        others = [item for item in parent2 if item not in child[a:b]]
        j = 0
        for i in range(self.Travelling_salesman.nums):
            if child[i] == -1:
                child[i] = others[j]
                j += 1
        return child
        
    def mutation(self, route):
        i, j = np.random.choice(len(route), 2, replace=False)
        route[i], route[j] = route[j], route[i]
        return route
        
    def birth_children(self):
        new_population = []
        fitnesses = {}
        for i in range(len(self.pop)):
            fitnesses[self.Travelling_salesman.route(self.Travelling_salesman.get_route(self.pop[i]))] = self.pop[i]
        sorted_fitnesses = dict(sorted(fitnesses.items()))
        parents = list(sorted_fitnesses.values())[0:self.top_best]
        route_best_parent = self.Travelling_salesman.route(self.Travelling_salesman.get_route(parents[0]))
        if route_best_parent < self.len_best_route:
            self.best_route = parents[0]
            self.len_best_route = route_best_parent
        for i in range(len(parents)):
            new_population.append(parents[i])
        pairs = [[parents[i], parents[j]] for i in range(len(parents)) for j in range(i + 1, len(parents))]
        for i in range(len(pairs)):
            child1 = self.crossover(pairs[i][0], pairs[i][1])
            child2 = self.crossover(pairs[i][1], pairs[i][0])
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

    def run(self):
        pop_num = 1
        while pop_num != self.pop_num_max:
            self.birth_children()
            pop_num += 1
        return self.best_route, self.len_best_route

    def plot_best(self):
        if not self.run_already:
            self.run()
            self.run_already = True
        plt.scatter([i[0] for i in self.Travelling_salesman.cities], [i[1] for i in self.Travelling_salesman.cities], c = 'orchid')
        route = self.Travelling_salesman.get_route(self.best_route)
        for i in range(self.Travelling_salesman.nums - 1):
            plt.plot([route[i][0], route[i+1][0]], [route[i][1], route[i+1][1]], linewidth = 2.0, c = '#1f77b4')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Found route with lenght {i}, best possible - {j}".format(i = self.len_best_route, j = self.Travelling_salesman.min_possible_route))
        plt.show()
        
    def plot_dynamic(self):
        if self.run_already:
            self.pop = self.create_first_population()
            self.best_route = self.pop[0]
            self.len_best_route = self.Travelling_salesman.route(self.Travelling_salesman.get_route(self.best_route))
        pop_num = 1
        while pop_num != self.pop_num_max:
            clear_output(wait=True)
            plt.scatter([i[0] for i in self.Travelling_salesman.cities], [i[1] for i in self.Travelling_salesman.cities], c = 'orchid')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Found route with lenght {i}, best possible - {j}".format(i = self.len_best_route, j = self.Travelling_salesman.min_possible_route))
            route = self.Travelling_salesman.get_route(self.best_route)
            for i in range(self.Travelling_salesman.nums - 1):
                plt.plot([route[i][0], route[i+1][0]], [route[i][1], route[i+1][1]], linewidth = 2.0, c = '#1f77b4')
            self.birth_children()
            pop_num += 1
            plt.show()
        self.run_already = True