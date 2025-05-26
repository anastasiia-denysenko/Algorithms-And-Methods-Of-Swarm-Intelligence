import random
import numpy as np
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from functions import *
np.seterr(divide='ignore', invalid='ignore')

class Bees:
    def __init__(self, func, delta, n_max, alpha, num_iter, size_pop, num_ar, num_el_ar, num_fl_el, num_fl, limitations, minimize=True):
        if not (0 < delta <= 1): raise ValueError(f"delta must be in (0,1], got {delta}")
        if not (0 < n_max <= 1): raise ValueError(f"n_max must be in (0,1], got {n_max}")
        if not (0 < alpha <= 1): raise ValueError(f"alpha must be in (0,1], got {alpha}")

        self.func = func
        self.delta = delta
        self.n_max = n_max
        self.alpha = alpha
        self.num_iter = num_iter
        self.size_pop = size_pop
        self.num_ar = num_ar
        self.num_el_ar = num_el_ar
        self.num_fl_el = num_fl_el
        self.num_fl = num_fl
        self.limitations = np.array(limitations)
        self.dim = len(limitations)
        self.minimize = minimize
        self.curr_iter = 0
        self.population = self.create_first_population()
        self.best_position, self.population = self.find_fitness_and_sort()
        self.history = [self.func(self.best_position)]

    def create_first_population(self):
        population = np.empty((self.size_pop, self.size_pop, self.dim))
        for d in range(self.dim):
            low, high = self.limitations[d]
            coords = low + (high - low) * np.random.rand(self.size_pop, self.size_pop)
            population[:, :, d] = coords
        return population

    def evaluate(self, position):
        return self.func(position)

    def find_fitness_and_sort(self):
        flat_pop = self.population.reshape(-1, self.dim)
        fitness = np.array([self.evaluate(ind) for ind in flat_pop])
        idx_sorted = np.argsort(fitness) if self.minimize else np.argsort(-fitness)
        sorted_flat_pop = flat_pop[idx_sorted]
        sorted_population = sorted_flat_pop.reshape(self.size_pop, self.size_pop, self.dim)
        best_position = sorted_population[0, 0]
        return best_position, sorted_population

    def iter(self):
        self.curr_iter += 1
        n_n = self.n_max * self.alpha ** self.curr_iter

        for i in range(self.num_ar):
            z = self.num_fl_el if i <= self.num_el_ar else self.num_fl
            for j in range(z):
                for k in range(z):
                    for d in range(self.dim):
                        low, high = self.limitations[d]
                        delta_val = n_n * self.delta * (high - low) * (-1 + 2 * random.random())
                        self.population[j, k, d] += delta_val
                        self.population[j, k, d] = np.clip(self.population[j, k, d], low, high)

        for m in range(self.num_ar, self.size_pop):
            for n in range(self.num_ar, self.size_pop):
                for d in range(self.dim):
                    low, high = self.limitations[d]
                    self.population[m, n, d] = low + (high - low) * random.random()

        self.best_position, self.population = self.find_fitness_and_sort()
        self.history.append(self.func(self.best_position))
        return self.population

    def run(self):
        for _ in range(self.num_iter):
            self.iter()
        return self.best_position
    
    def plot_values(self):
        self.run()
        plt.plot([i for i in range(len(self.history))], self.history)
        plt.xlabel('Iterations')
        plt.ylabel('Function values')
        plt.title('Best value found: {V}'.format(V = self.func(self.best_position)))
        plt.grid()
        plt.show()
        
    def plot(self):
        if self.dim != 2:
            raise ValueError("Visualization only supported for 2D problems.")

        self.run()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X = np.linspace(self.limitations[0][0], self.limitations[0][1], 200)
        Y = np.linspace(self.limitations[1][0], self.limitations[1][1], 200)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([[self.func(np.array([x, y])) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.viridis, linewidth=0, antialiased=False, alpha=0.25)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')

        best_x, best_y = self.best_position
        best_z = self.func(self.best_position)
        ax.scatter(best_x, best_y, best_z, c='orchid')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def run_and_plot(self):
        if self.dim != 2:
            raise ValueError("Visualization only supported for 2D problems.")

        X = np.linspace(self.limitations[0][0], self.limitations[0][1], 200)
        Y = np.linspace(self.limitations[1][0], self.limitations[1][1], 200)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([[self.func(np.array([x, y])) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(*self.limitations[0])
        ax.set_ylim(*self.limitations[1])
        ax.set_zlim(np.nanmin(Z), np.nanmax(Z))

        surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.viridis, alpha=0.3)

        pop_scat = ax.scatter([], [], [], c='cornflowerblue')
        best_scat = ax.scatter([], [], [], c='orchid')

        def update(frame):
            self.iter()

            flat_pop = self.population.reshape(-1, self.dim)
            xs = flat_pop[:, 0]
            ys = flat_pop[:, 1]
            zs = np.array([self.func(p) for p in flat_pop])

            pop_scat._offsets3d = (xs, ys, zs)

            best_x, best_y = self.best_position
            best_z = self.func(self.best_position)
            best_scat._offsets3d = ([best_x], [best_y], [best_z])

            ax.set_title(f"Bees Iteration {frame + 1}")

        ani = matplotlib.animation.FuncAnimation(fig, update, frames=self.num_iter, interval=100, repeat=False)
        plt.show()
    
#beexy2 = Bees(Rosenbrock, 0.15, 0.34, 0.72, 50, 10, 3, 10, 5, 2, [[-2, 2], [-2, 2]])
#beexy2.plot()
#beexy2 = Bees(rastrigin, 0.15, 0.34, 0.72, 50, 10, 3, 10, 5, 2, [[-5.12, 5.12] for _ in range(15)])
#beexy2.plot_values()