import random
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.ticker import LinearLocator
from copy import deepcopy

class Fireflies:
    def __init__(self, func, beta_max, gamma, alpha, max_iter, size_pop, x_min, x_max, y_min, y_max, minimize = True):
        if beta_max <= 0 or beta_max >= 1:
            raise ValueError("Beta_max has to be in range (0, 1), but given value is {i}".format(i = beta_max))
        if gamma <= 0 or gamma >= 1:
            raise ValueError("Gamma has to be in range (0, 1), but given value is {i}".format(i = gamma))
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha has to be in range [0, 1], but given value is {i}".format(i = alpha))
        self.func = func
        self.beta_max = beta_max
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.size_pop = size_pop
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.minimize = minimize
        self.pop_x, self.pop_y = self.create_first_population()
        self.curr_iter = 0
        self.x_ideal = self.pop_x[0][0]
        self.y_ideal = self.pop_y[0][0]
    def create_first_population(self):
        p_x, p_y = [], []
        for i in range(self.size_pop):
            p_x.append(self.x_min + (self.x_max - self.x_min)*random.uniform(0, 1))
            p_y.append(self.y_min + (self.y_max - self.y_min)*random.uniform(0, 1))
        self.pop_x = np.array(p_x)
        self.pop_y = np.array(p_y)
        self.pop_x, self.pop_y = np.meshgrid(self.pop_x, self.pop_y)
        return self.pop_x, self.pop_y
    def iter(self):
        for i in range(self.size_pop):
            for j in range(self.size_pop):
                if self.func(self.pop_x[i][j], self.y_ideal) < self.func(self.x_ideal, self.y_ideal):
                    self.x_ideal = self.pop_x[i][j]
                if self.func(self.x_ideal, self.pop_y[i][j]) < self.func(self.x_ideal, self.y_ideal):
                    self.y_ideal = self.pop_y[i][j]
                for k in range(self.size_pop):
                    for m in range(self.size_pop):
                        if self.func(self.pop_x[i][j], self.y_ideal) < self.func(self.pop_x[k][m], self.y_ideal):
                            beta = self.beta_max*np.e**(-self.gamma*(self.pop_x[i][j] - self.pop_x[k][m])**2)
                            self.pop_x[k][m] += beta*(self.pop_x[i][j] - self.pop_x[k][m]) + self.alpha*(random.uniform(0, 1) - 0.5)
                            if self.pop_x[k][m] < self.x_min:
                                self.pop_x[k][m] = self.x_min
                            elif self.pop_x[k][m] > self.x_max:
                                self.pop_x[k][m] = self.x_max
                        if self.func(self.x_ideal, self.pop_y[i][j]) < self.func(self.x_ideal, self.pop_y[k][m]):
                            beta = self.beta_max*np.e**(-self.gamma*(self.pop_y[i][j] - self.pop_y[k][m])**2)
                            self.pop_y[k][m] += beta*(self.pop_y[i][j] - self.pop_y[k][m]) + self.alpha*(random.uniform(0, 1) - 0.5)
                            if self.pop_y[k][m] < self.y_min:
                                self.pop_y[k][m] = self.y_min
                            elif self.pop_y[k][m] > self.y_max:
                                self.pop_y[k][m] = self.y_max
        self.curr_iter += 1
        return self.pop_x, self.pop_y, self.x_ideal, self.y_ideal
    def run(self):
        while self.curr_iter != self.max_iter:
            self.iter()
        return self.pop_x, self.pop_y, self.x_ideal, self.y_ideal
    def plot(self):
        self.run()
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        X, Y = np.arange(self.x_min, self.x_max, 0.01), np.arange(self.y_min, self.y_max, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.viridis, linewidth=0, antialiased=False, alpha=0.25)
        ax.set_zlim(Z.min(), Z.max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.scatter(self.x_ideal, self.y_ideal, self.func(self.x_ideal, self.y_ideal), c = 'orchid')
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
            best._offsets3d = ([self.x_ideal], [self.y_ideal], [self.func(self.x_ideal, self.y_ideal)])
            title.set_text('Fireflies, iteration {}'.format(num+1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('Fireflies')
        all_pop = ax.scatter(self.pop_x[0][0], self.pop_y[0][0], self.func(self.pop_x[0][0], self.pop_y[0][0]), c = 'cornflowerblue', zorder = 1)
        best = ax.scatter([self.x_min], [self.y_min], [self.func(self.x_min, self.y_min)], c = 'orchid', zorder = 1)
        surf = ax.plot_surface(X, Y, Z, cmap = matplotlib.cm.viridis, linewidth=0, antialiased = False, alpha=0.25)
        ani = matplotlib.animation.FuncAnimation(fig, update_graph, self.max_iter, interval=60, blit=False, repeat=False)
        ani.save(str(random.uniform(100000, 900000))+'.gif', dpi=300, writer=PillowWriter(fps=25))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axes.set_xlim3d(left=self.x_min, right=self.x_max)
        ax.axes.set_ylim3d(bottom=self.y_min, top=self.y_max) 
        plt.show()