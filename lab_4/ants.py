import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random

class Ant:
    def __init__(self, curr_city, num_cities):
        self.curr_city = curr_city
        self.num_cities = num_cities
        self.route = [curr_city]
        self.not_visited = [i for i in range(self.num_cities) if i != curr_city]
    def move(self, new_city):
        self.curr_city = new_city
        ind = self.not_visited.index(new_city)
        self.not_visited.remove(new_city)
        self.route.append(new_city)
        return ind
        
class Ants:
    def __init__(self, Travelling_salesman, size_pop, num_iter, alpha, beta, q, rho):
        self.Travelling_salesman = Travelling_salesman
        self.size_pop = size_pop
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.rho = rho
        self.vision = [[1/j if j != 0 else 0 for j in i] for i in self.Travelling_salesman.distance]
        self.pheromone = [[0.1 for i in range(self.Travelling_salesman.nums)] for i in range(self.Travelling_salesman.nums)]
        self.probs = [[1/self.Travelling_salesman.nums for j in range(self.Travelling_salesman.nums-1)] for i in range(self.size_pop)]
        self.pop = self.create_ants()
        self.best_len = float('inf')
        self.best_route = random.sample([i for i in range(self.Travelling_salesman.nums)], self.Travelling_salesman.nums)

    def create_ants(self):
        pop = []
        cities = [i for i in range(self.Travelling_salesman.nums)]
        for _ in range(self.size_pop):
            pop.append(Ant(np.random.choice(cities), self.Travelling_salesman.nums))
        return pop

    def reset_parameters(self):
        self.vision = [[1/j if j != 0 else 0 for j in i] for i in self.Travelling_salesman.distance]
        self.pheromone = [[0.1 for i in range(self.Travelling_salesman.nums)] for i in range(self.Travelling_salesman.nums)]
        self.probs = [[1/self.Travelling_salesman.nums for j in range(self.Travelling_salesman.nums-1)] for i in range(self.size_pop)]
        self.pop = self.create_ants()

    def update_pheromone(self):
        for i in range(self.Travelling_salesman.nums):
            for j in range(self.Travelling_salesman.nums):
                self.pheromone[i][j] *= (1 - self.rho)
                self.pheromone[i][j] += self.q / self.Travelling_salesman.distance[i][j] if self.Travelling_salesman.distance[i][j] != 0 else 0

    def update_probabilities(self):
        for i in range(self.size_pop):
            for j in range(len(self.probs[0])):
                self.probs[i][j] = (self.pheromone[self.pop[i].curr_city][j]**self.alpha * self.vision[self.pop[i].curr_city][j]**self.beta)/sum(sum(i) for i in self.probs)
            s = sum(self.probs[i])
            if s > 1 or s < 1:
                diff = 1/s
                for j in range(len(self.probs[i])):
                    self.probs[i][j] *= diff

    def go_to_next_city(self):
        for i in range(self.size_pop):
            go_to = np.random.choice(self.pop[i].not_visited, p = self.probs[i])
            ind = self.pop[i].move(go_to)
            del self.probs[i][ind]

    def iter(self):
        while len(self.pop[0].not_visited) != 1:
            self.update_pheromone()
            self.update_probabilities()
            self.go_to_next_city()
        for i in range(self.size_pop):
            self.pop[i].move(self.pop[i].not_visited[0])
            curr_route_len = self.Travelling_salesman.route(self.Travelling_salesman.get_route(self.pop[i].route))
            if curr_route_len < self.best_len:
                self.best_len = curr_route_len
                self.best_route = self.pop[i].route
        return self.best_len, self.best_route

    def run(self):
        for _ in range(self.num_iter):
            self.best_len, self.best_route = self.iter()
            self.reset_parameters()

    def plot_best(self):
        self.run()
        plt.scatter([i[0] for i in self.Travelling_salesman.cities], [i[1] for i in self.Travelling_salesman.cities], c = 'orchid')
        route = self.Travelling_salesman.get_route(self.best_route)
        for i in range(self.Travelling_salesman.nums - 1):
            plt.plot([route[i][0], route[i+1][0]], [route[i][1], route[i+1][1]], linewidth = 2.0, c = '#1f77b4')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Found route with lenght {i}, best possible - {j}".format(i = self.best_len, j = self.Travelling_salesman.min_possible_route))
        plt.show()

    def plot_dynamic(self):
        iters = 0
        while iters != self.num_iter:
            clear_output(wait=True)
            plt.scatter([i[0] for i in self.Travelling_salesman.cities], [i[1] for i in self.Travelling_salesman.cities], c = 'orchid')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Found route with lenght {i}, best possible - {j}".format(i = self.best_len, j = self.Travelling_salesman.min_possible_route))
            route = self.Travelling_salesman.get_route(self.best_route)
            for i in range(self.Travelling_salesman.nums - 1):
                plt.plot([route[i][0], route[i+1][0]], [route[i][1], route[i+1][1]], linewidth = 2.0, c = '#1f77b4')
            self.best_len, self.best_route = self.iter()
            self.reset_parameters()
            iters += 1
            plt.show()