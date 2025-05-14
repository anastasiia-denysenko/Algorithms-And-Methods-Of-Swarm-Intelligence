import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class Genetic_optimization_for_x:
    def __init__(self, func, pop_num_max, top_best, x_min, x_max, chrom_lenght, prob_mutation, pop_size, minimize = True):
        self.func = func
        self.pop_num_max = pop_num_max
        self.top_best = top_best
        self.x_min = x_min
        self.x_max = x_max
        self.chrom_lenght = chrom_lenght
        self.prob_mutation = prob_mutation
        self.pop_size = pop_size
        self.minimize = minimize
        self.pop = self.create_first_population()
        self.best = self.pop[0]
        self.best_fitness = self.fitness(self.best)

    def create_first_population(self):
        f_pop = []
        for i in range(self.pop_size):
            chromosome = ""
            for j in range(self.chrom_lenght):
                chromosome += str(random.randrange(0, 2))
            f_pop.append(chromosome)
        return f_pop
    
    def bin_to_dec_a_to_b(self, chromosome):
        return int(chromosome, 2)*(self.x_max - self.x_min)/(2**self.chrom_lenght - 1) + self.x_min
    
    def crossover(self, p1, p2):
        divide_by = random.randrange(1, self.chrom_lenght)
        child1, child2 = p1[:divide_by] + p2[divide_by:], p2[:divide_by] + p1[divide_by:]
        return child1, child2
    
    def mutation(self, chromosome):
        gene = random.randrange(0, self.chrom_lenght)
        chromosome = chromosome[:gene] + '0' + chromosome[gene+1:] if chromosome[gene] == '1' else chromosome[:gene] + '0' + chromosome[gene+1:]
        return chromosome
    
    def fitness(self, chromosome):
        fit = self.func(self.bin_to_dec_a_to_b(chromosome))
        return fit
    
    def birth_children(self):
        new_population = []
        fitnesses = {}
        for i in range(len(self.pop)):
            fitnesses[self.fitness(self.pop[i])] = self.pop[i]
        sorted_fitnesses = dict(sorted(fitnesses.items()))
        if self.minimize == True:
            parents = list(sorted_fitnesses.values())[0:self.top_best]
        else:
            parents = list(sorted_fitnesses.values())[-self.top_best:]
        for i in range(len(parents)):
            new_population.append(parents[i])
        pairs = [[parents[i], parents[j]] for i in range(len(parents)) for j in range(i + 1, len(parents))]
        for i in range(len(pairs)):
            child1, child2 = self.crossover(pairs[i][0], pairs[i][0])
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
    
    def iter(self):
        self.birth_children()
        if self.minimize == True:
            for i in range(len(self.pop)):
                if self.fitness(self.pop[i]) < self.best_fitness:
                    self.best = self.pop[i]
                    self.best_fitness = self.fitness(self.best)
        else:
            for i in range(len(self.pop)):
                if self.fitness(self.pop[i]) > self.best_fitness:
                    self.best = self.pop[i]
                    self.best_fitness = self.fitness(self.best)
    
    def run(self):
        for _ in range(self.pop_num_max):
            self.iter()
    
    def plot(self):
        self.run()
        xs = np.linspace(self.x_min, self.x_max, num = 350)
        f = self.func(xs)
        fig, ax = plt.subplots()
        ax.plot(xs, f, linewidth = 2.0)
        ax.scatter(self.bin_to_dec_a_to_b(self.best), self.best_fitness, c = 'orchid')
        plt.show()

    def plot_dynamic(self):
        xs = np.linspace(self.x_min, self.x_max, num = 350)
        f = self.func(xs)
        for _ in range(self.pop_num_max):
            clear_output(wait=True)
            plt.plot(xs, f, linewidth = 2.0, c = '#1f77b4')
            self.iter()
            dec = [self.bin_to_dec_a_to_b(i) for i in self.pop]
            func = [self.func(i) for i in dec]
            plt.scatter(dec, func, c = 'lightblue')
            plt.scatter(self.bin_to_dec_a_to_b(self.best), self.best_fitness, c = 'orchid')
            plt.show()


class Genetic_optimization_for_x_and_y:
    def __init__(self, func, pop_num_max, top_best, x_min, x_max, y_min, y_max, chrom_lenght, prob_mutation, minimize = True):
        self.func = func
        self.pop_num_max = pop_num_max
        self.top_best = top_best
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.chrom_lenght = chrom_lenght
        self.prob_mutation = prob_mutation
        self.minimize = minimize
        self.population_x, self.population_y = self.create_first_population(), self.create_first_population()

    def create_first_population(self):
        f_pop = []
        for i in range(self.top_best**2):
            chromosome = ""
            for j in range(self.chrom_lenght):
                chromosome += str(random.randrange(0, 2))
            f_pop.append(chromosome)
        return f_pop
    
    def bin_to_dec_a_to_b(self, chromosome, x = True):
        if x == True:
            return int(chromosome, 2)*(self.x_max - self.x_min)/(2**self.chrom_lenght - 1) + self.x_min
        else:
            return int(chromosome, 2)*(self.y_max - self.y_min)/(2**self.chrom_lenght - 1) + self.y_min
        
    def fitness_x_and_y(self, chromosome_x, chromosome_y):
        fit = self.func(self.bin_to_dec_a_to_b(chromosome_x, True), self.bin_to_dec_a_to_b(chromosome_y, False))
        return fit
    
    def crossover(self, p1, p2):
        divide_by = random.randrange(1, self.chrom_lenght)
        child1, child2 = p1[:divide_by] + p2[divide_by:], p2[:divide_by] + p1[divide_by:]
        return child1, child2
    
    def mutation(self, chromosome):
        gene = random.randrange(0, self.chrom_lenght)
        chromosome = chromosome[:gene] + '0' + chromosome[gene+1:] if chromosome[gene] == '1' else chromosome[:gene] + '0' + chromosome[gene+1:]
        return chromosome
    
    def birth_children(self, population_x, population_y):
        new_population = []
        fitnesses = {}
        for i in range(min(len(population_x), len(population_y))):
            fitnesses[self.fitness_x_and_y(population_x[i], population_y[i])] = population_x[i]
        sorted_fitnesses = dict(sorted(fitnesses.items()))
        if self.minimize == True:
            parents = list(sorted_fitnesses.values())[0:self.top_best]
        else:
            parents = list(sorted_fitnesses.values())[-self.top_best:]
        for i in range(len(parents)):
            new_population.append(parents[i])
        pairs = [[parents[i], parents[j]] for i in range(len(parents)) for j in range(i + 1, len(parents))]
        for i in range(len(pairs)):
            child1, child2 = self.crossover(pairs[i][0], pairs[i][1])
            chance = random.uniform(0, 1)
            child = random.randrange(0, 1)
            if chance <= self.prob_mutation:
                if child == 0:
                    child1 = self.mutation(child1)
                else:
                    child2 = self.mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        return new_population

    def iter(self):
        self.population_x, self.population_y = self.birth_children(self.population_x, self.population_y), self.birth_children(self.population_y, self.population_x)
        dec_x, dec_y, dec_z = [], [], []
        for i in range(len(self.population_x)):
            dec_x.append(self.bin_to_dec_a_to_b(self.population_x[i], True))
            dec_y.append(self.bin_to_dec_a_to_b(self.population_y[i], False))
            dec_z.append(self.func(dec_x[-1], dec_y[-1]))
        best_x, best_y, best_z = dec_x[0], dec_y[0], dec_z[0]
        if self.minimize == True:
            for i in range(len(dec_x)):
                for j in range(len(dec_y)):
                    if self.func(dec_x[i], dec_y[j]) <= self.func(best_x, best_y):
                        best_x, best_y, best_z = dec_x[i], dec_y[j], self.func(dec_x[i], dec_y[j])
        else:
            for i in range(len(dec_x)):
                for j in range(len(dec_y)):
                    if self.func(dec_x[i], dec_y[j]) >= self.func(best_x, best_y):
                        best_x, best_y, best_z = dec_x[i], dec_y[j], self.func(dec_x[i], dec_y[j])
        return self.population_x, self.population_y, dec_x, dec_y, dec_z, best_x, best_y, best_z
    
    def run(self):
        for _ in range(self.pop_num_max):
            self.population_x, self.population_y, dec_x, dec_y, dec_z, best_x, best_y, best_z = self.iter()
        return dec_x, dec_y, dec_z, best_x, best_y, best_z
    
    def plot_x_and_y(self):
        _, _, _, x, y, z = self.run()
        fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
        X, Y = np.arange(self.x_min, self.x_max, 0.25), np.arange(self.y_min, self.y_max, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.25)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(Z.min(),Z.max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.scatter(x, y, z, c = 'orchid')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        return x, y, z
    
    def plot_dynamic(self):
        population_x, population_y = self.create_first_population(), self.create_first_population()
        X, Y = np.arange(self.x_min, self.x_max, 0.25), np.arange(self.y_min, self.y_max, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        def update_graph(num):
            self.population_x, self.population_y, dec_x, dec_y, dec_z, best_x, best_y, best_z = self.iter()
            all_pop._offsets3d = (dec_x, dec_y, dec_z)
            best._offsets3d = ([best_x], [best_y], [best_z])
            title.set_text('Genetic optimization, iteration {}'.format(num+1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('Genetic optimization')
        all_pop = ax.scatter([], [], [], c = 'cornflowerblue', zorder = 1)
        best = ax.scatter([], [], [], c = 'orchid', zorder = 1)
        surf = ax.plot_surface(X, Y, Z, cmap = cm.viridis, linewidth=0, antialiased = False, alpha=0.25)
        ani = animation.FuncAnimation(fig, update_graph, self.pop_num_max, interval=60, blit=False, repeat=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()