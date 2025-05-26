import random
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.ticker import LinearLocator
from copy import deepcopy
from functions import *

class Fireflies:
    def __init__(self, func, beta_max, gamma, alpha, max_iter, size_pop, limitations, minimize=True):
        if not (0 < beta_max < 1): raise ValueError(f"beta_max must be in (0, 1), got {beta_max}")
        if not (0 < gamma < 1): raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if not (0 <= alpha <= 1): raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.func = func
        self.beta_max = beta_max
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.size_pop = size_pop
        self.limitations = np.array(limitations)
        self.dim = len(self.limitations)
        self.minimize = minimize
        self.population = self.create_first_population()
        self.curr_iter = 0
        self.best_position = self.population[0, 0]
        self.best_value = self.func(self.best_position)
        self.history = [self.best_value]

    def create_first_population(self):
        population = np.empty((self.size_pop, self.size_pop, self.dim))
        for d in range(self.dim):
            low, high = self.limitations[d]
            coords = low + (high - low) * np.random.rand(self.size_pop, self.size_pop)
            population[:, :, d] = coords
        return population

    def brightness(self, position):
        val = self.func(position)
        return -val if self.minimize else val

    def iter(self):
        for i in range(self.size_pop):
            for j in range(self.size_pop):
                pos_ij = self.population[i, j]
                bright_ij = self.brightness(pos_ij)

                if (bright_ij > self.brightness(self.best_position)):
                    self.best_position = pos_ij.copy()
                    self.best_value = self.func(self.best_position)

                for k in range(self.size_pop):
                    for m in range(self.size_pop):
                        pos_km = self.population[k, m]
                        bright_km = self.brightness(pos_km)

                        if bright_ij > bright_km:
                            dist_sq = np.sum((pos_ij - pos_km) ** 2)
                            beta = self.beta_max * np.exp(-self.gamma * dist_sq)
                            rand = self.alpha * (np.random.rand(self.dim) - 0.5)

                            self.population[k, m] += beta * (pos_ij - pos_km) + rand

                            for d in range(self.dim):
                                low, high = self.limitations[d]
                                self.population[k, m, d] = np.clip(self.population[k, m, d], low, high)

        self.curr_iter += 1
        return self.population, self.best_position

    def run(self):
        while self.curr_iter < self.max_iter:
            self.iter()
            self.history.append(self.best_value)
        return self.population, self.best_position
    
    def plot_score(self):
        self.run()
        plt.plot([i for i in range(len(self.history))], self.history)
        plt.xlabel('Iterations')
        plt.ylabel('Function values')
        plt.title('Best value found: {V}'.format(V = self.history[-1]))
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
        ax.scatter(self.best_position[1], self.best_position[0], self.func(self.best_position[1], self.best_position[0]), color='orchid')
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
        ax.set_zlim(np.nanmin(Z), np.nanmax(Z)) if not np.isinf(np.nanmax(Z)) else ax.set_zlim(np.nanmin(Z), 500)
        surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.viridis, alpha=0.3)
        pop_scat = ax.scatter([], [], [], c='cornflowerblue')
        best_scat = ax.scatter([], [], [], c='orchid')

        def update(frame):
            self.population, self.best_position = self.iter()
            flat_pop = self.population.reshape(-1, self.dim)
            xs = flat_pop[:, 0]
            ys = flat_pop[:, 1]
            zs = np.array([self.func(p) for p in flat_pop])

            pop_scat._offsets3d = (xs, ys, zs)

            best_x, best_y = self.best_position
            best_z = self.func(self.best_position)
            best_scat._offsets3d = ([best_x], [best_y], [best_z])

            ax.set_title(f"Fireflies Iteration {frame + 1}")

        ani = matplotlib.animation.FuncAnimation(fig, update, frames=self.max_iter, interval=100, repeat=False)
        #ani.save(str(random.uniform(100000, 900000))+'.gif', dpi=300, writer=PillowWriter(fps=25))
        plt.show()

f = Fireflies(mishra_bird, 0.2, 0.34, 1, 25, 5, [[-10, 0], [-6.5, 0]], minimize = True)
#f.plot()
#f = Fireflies(Rosenbrock, 0.2, 0.34, 1, 25, 5, [[-1.5, 1.5], [-1.5, 1.5]], minimize = True)
#f.run_and_plot()
