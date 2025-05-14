import random
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.ticker import LinearLocator
from copy import deepcopy

class Bees:
    def __init__(self, func, delta, n_max, alpha, num_iter, size_pop, num_ar, num_el_ar, num_fl_el, num_fl, x_min, x_max, y_min, y_max, minimize = True):
        if delta > 1 or delta < 0:
            raise ValueError("Value for delta gas to be in range (0, 1), but given value is {i}".format(i = delta))
        if n_max > 1 or n_max < 0:
            raise ValueError("Value for n_max gas to be in range (0, 1), but given value is {i}".format(i = n_max))
        if alpha > 1 or alpha < 0:
            raise ValueError("Value for alpha gas to be in range (0, 1), but given value is {i}".format(i = alpha))
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
        self.x_min =  x_min
        self.x_max = x_max
        self.y_min =  y_min
        self.y_max = y_max
        self.minimize = minimize
        self.pop_x, self.pop_y = self.create_first_population()
        self.x_perfect, self.y_perfect, _,  _ = self.find_fitness_and_sort()
        self.curr_iter = 0
    def create_first_population(self):
        p_x, p_y = [], []
        for i in range(self.size_pop):
            p_x.append(self.x_min + (self.x_max - self.x_min)*random.uniform(0, 1))
            p_y.append(self.y_min + (self.y_max - self.y_min)*random.uniform(0, 1))
        self.pop_x = np.array(p_x)
        self.pop_y = np.array(p_y)
        self.pop_x, self.pop_y = np.meshgrid(self.pop_x, self.pop_y)
        return self.pop_x, self.pop_y
    def find_fitness_and_sort(self):
        pop_x = {}
        pop_y = {}
        for i in range(self.size_pop):
            for j in range(self.size_pop):
                pop_x[self.pop_x[i][j]] = self.func(self.pop_x[i][j], self.pop_y[i][j])
                pop_y[self.pop_y[i][j]] = self.func(self.pop_x[i][j], self.pop_y[i][j])
        pop_x = sorted(pop_x.items(), key = lambda x:x[1], reverse = not self.minimize)
        pop_y = sorted(pop_y.items(), key = lambda x:x[1], reverse = not self.minimize)
        res_x = [ item[0] for item in pop_x ]
        res_y = [ item[0] for item in pop_y ]
        self.pop_x = np.array(res_x)
        self.pop_y = np.array(res_y)
        self.pop_x, self.pop_y = np.meshgrid(self.pop_x, self.pop_y)
        self.x_perfect =  self.pop_x[0][0]
        self.y_perfect =  self.pop_y[0][0]
        return self.x_perfect, self.y_perfect, self.pop_x, self.pop_y
    def iter(self):
        self.curr_iter += 1
        for i in range(self.num_ar): 
            n_n = self.n_max*self.alpha**self.curr_iter
            if i <= self.num_el_ar:
                z = self.num_fl_el
            else:
                z = self.num_fl
            for j in range(z):
                for k in range(z):
                    self.pop_x[j][k] += n_n*self.delta*(self.x_max - self.x_min)*(-1+2*random.uniform(0, 1))
                    self.pop_y[j][k] += n_n*self.delta*(self.y_max - self.y_min)*(-1+2*random.uniform(0, 1))
                    if self.pop_x[j][k] > self.x_max:
                        self.pop_x[j][k] = self.x_max
                    elif self.pop_x[j][k] < self.x_min:
                        self.pop_x[j][k] = self.x_min
                    if self.pop_y[j][k] > self.y_max:
                        self.pop_y[j][k] = self.y_max
                    elif self.pop_y[j][k] < self.y_min:
                        self.pop_y[j][k] = self.y_min
        for m in range(self.num_ar, self.size_pop):
            for n in range(self.num_ar, self.size_pop):
                self.pop_x[m][n] = self.x_min + (self.x_max - self.x_min)*random.uniform(0, 1)
                self.pop_y[m][n] = self.y_min + (self.y_max - self.y_min)*random.uniform(0, 1)
        return self.pop_x, self.pop_y
    def run(self):
        for _ in range(self.num_iter):
            self.iter()
            self.find_fitness_and_sort()
        return self.x_perfect, self.y_perfect
    def plot(self):
        self.run()
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        X, Y = np.arange(self.x_min, self.x_max, 0.01), np.arange(self.y_min, self.y_max, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.viridis, linewidth=0, antialiased=False, alpha=0.25)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.scatter(self.x_perfect, self.y_perfect, self.func(self.x_perfect, self.y_perfect), c = 'orchid')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    def run_and_plot(self):
        X, Y = np.arange(self.x_min, self.x_max, 0.01), np.arange(self.y_min, self.y_max, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        def update_graph(num):
            self.iter()
            for i in range(self.size_pop):
                pop_z = []
                for j in range(self.size_pop):
                    pop_z.append(self.func(self.pop_x[i][j], self.pop_y[i][j]))
                all_pop._offsets3d = (self.pop_x[i], self.pop_y[i], pop_z)
            best._offsets3d = ([self.x_perfect], [self.y_perfect], [self.func(self.x_perfect, self.y_perfect)])
            title.set_text('Bees, iteration {}'.format(num+1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('Bees')
        all_pop = ax.scatter(self.pop_x[0], self.pop_y[0], self.func(self.pop_x, self.pop_y)[0], c = 'cornflowerblue', zorder = 1)
        best = ax.scatter([self.x_min], [self.y_min], [self.func(self.x_min, self.y_min)], c = 'orchid', zorder = 1)
        surf = ax.plot_surface(X, Y, Z, cmap = matplotlib.cm.viridis, linewidth=0, antialiased = False, alpha=0.25)
        ani = matplotlib.animation.FuncAnimation(fig, update_graph, self.num_iter, interval=60, blit=False, repeat=False)
        ani.save(str(random.uniform(100000, 900000))+'.gif', dpi=300, writer=PillowWriter(fps=25))   
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axes.set_xlim3d(left=self.x_min, right=self.x_max)
        ax.axes.set_ylim3d(bottom=self.y_min, top=self.y_max) 
        plt.show()