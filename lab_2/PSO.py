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

class PSO(object):
    def __init__(self, func, num_iter, a1, a2, size_pop, v_min, v_max, limitations, minimize = True):
        if a1 < 0 or a1 > 4 or a2 < 0 or a2 > 4:
            raise ValueError("Accelerations has to be in range (0, 4), but given values are {i} and {j}".format(i = a1, j = a2))
        if v_max <= 0:
            raise ValueError("Maximum velocity has to be greater than 0, but given value is {i}".format(i = v_max))
        self.func = func
        self.num_iter = num_iter
        self.a1 = a1
        self.a2 = a2
        self.size_pop = size_pop
        self.v_min = v_min
        self.v_max = v_max
        self.limitations = limitations
        self.dims = len(self.limitations)
        self.minimize = minimize
        self.curr_iter = 0
        self.pop = self.create_first_population()
        self.bests = deepcopy(self.pop)
        self.ideal = [lim[0] for lim in self.limitations]
        self.perfect_score = self.func(self.ideal)
        self.vel = [[self.v_min + (self.v_max - self.v_min)*random.uniform(0, 1) for _ in range(self.dims)] for _ in range(self.size_pop)]
        self.history = [self.perfect_score]

    def create_first_population(self):
        population = []
        for _ in range(self.size_pop):
            individual = [np.random.uniform(low=lim[0], high=lim[1]) for lim in self.limitations]
            population.append(individual)
        return population
    
    def find_fitness(self):
        if self.minimize == True:
            for i in range(self.size_pop):
                if self.func(self.pop[i]) < self.func(self.bests[i]):
                    self.bests[i] = self.pop[i]
                    if self.func(self.pop[i]) < self.perfect_score:
                        self.ideal[:] = self.pop[i]
                        self.perfect_score = self.func(self.pop[i])
        else:
            for i in range(self.size_pop):
                if self.func(self.pop[i]) > self.func(self.bests[i]):
                    self.bests[i] = self.pop[i]
                    if self.func(self.pop[i]) > self.perfect_score:
                        self.ideal[:] = self.pop[i]
                        self.perfect_score = self.func(self.pop[i])
        return self.bests, self.ideal, self.perfect_score
    
    def create_velocity_vector(self):
        r1 = np.random.rand()
        r2 = np.random.rand()
        for i in range(self.size_pop):
            for j in range(self.dims):
                self.vel[i][j] += self.a1*(self.bests[i][j]-self.pop[i][j])*r1 + self.a2*(self.ideal[j] - self.pop[i][j])*r2
                self.vel[i][j] = np.clip(self.vel[i][j], self.v_min, self.v_max)
        return self.vel
    
    def update_positions(self):
        for i in range(self.size_pop):
            for j in range(self.dims):
                self.pop[i][j] += self.vel[i][j]
                self.pop[i][j] = np.clip(self.pop[i][j], self.limitations[j][0], self.limitations[j][1])
        return self.pop
    
    def iter(self):
        self.bests, self.ideal, self.perfect_score = self.find_fitness()
        self.history.append(self.perfect_score)
        self.vel = self.create_velocity_vector()
        self.pop = self.update_positions()
        self.curr_iter += 1
        return self.pop, self.ideal, self.perfect_score

    def run(self):
        while self.curr_iter != self.num_iter:
            self.iter()
        return self.pop, self.ideal, self.perfect_score
    
    def plot_score(self):
        self.run()
        plt.plot([i for i in range(len(self.history))], self.history)
        plt.xlabel('Iterations')
        plt.ylabel('Function values')
        plt.title('Best value found: {V}'.format(V = self.perfect_score))
        plt.grid()
        plt.show()

    def plot(self):
        if self.dims != 2:
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
        ax.scatter(self.ideal[0], self.ideal[1], self.perfect_score, c = 'orchid')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def run_and_plot(self):
        if self.dims != 2:
            raise ValueError("Visualization only supported for 2D problems.")

        X = np.linspace(self.limitations[0][0], self.limitations[0][1], 200)
        Y = np.linspace(self.limitations[1][0], self.limitations[1][1], 200)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([[self.func([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(*self.limitations[0])
        ax.set_ylim(*self.limitations[1])
        ax.set_zlim(Z.min(), Z.max()) if not np.isnan(Z.max) else ax.set_zlim(Z.min(), 500)

        surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.viridis, alpha=0.3)

        pop_scat = ax.scatter([], [], [], c = 'cornflowerblue')
        best_scat = ax.scatter([], [], [], c = 'orchid')

        def update(frame):
            self.pop, self.ideal[:], self.perfect_score = self.iter()
            xs = [p[0] for p in self.pop]
            ys = [p[1] for p in self.pop]
            zs = [self.func(p) for p in self.pop]

            pop_scat._offsets3d = (xs, ys, zs)
            best_scat._offsets3d = ([self.ideal[0]], [self.ideal[1]], [self.perfect_score])
            ax.set_title(f"PSO Iteration {frame + 1}")

        ani = matplotlib.animation.FuncAnimation(fig, update, frames=self.num_iter, interval=100, repeat=False)
        #ani.save(str(random.uniform(100000, 900000))+'.gif', dpi=300, writer=PillowWriter(fps=25))
        plt.show()

#psoxy = PSO(mishra_bird, 80, 2, 2, 20, -1, 3,  [[-10, 0], [-6.5, 0]])
#psoxy.plot()
#psoxy = PSO(rastrigin, 100, 2, 2, 20, -1, 3,  [[-5.12, 5.12] for _ in range(10)])
#psoxy = PSO(rastrigin, 100, 2, 2, 30, -1, 3,  [[-5.12, 5.12] for _ in range(2)])
#psoxy.plot_score()
#psoxy = PSO(mishra_bird, 100, 2, 2, 30, -1, 3, [[-10, 0], [-6.5, 0]])
#psoxy.run_and_plot()