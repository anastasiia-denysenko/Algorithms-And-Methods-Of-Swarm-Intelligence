import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class Wolves_optimization:
    def __init__(self, func, max_iter, s, step_x, x_min, x_max, minimize = True):
        self.func = func
        self.max_iter = max_iter
        self.s = s
        self.step_x = step_x
        self.x_min = x_min
        self.x_max = x_max
        self.minimize = minimize
        self.wolves = np.linspace(self.x_min, self.x_max, num = self.s)
        self.Gbest = self.func(self.wolves[0])
        self.x_wolf = self.wolves[0]

    def iter(self):
        for wolf in range(len(self.wolves)):
            if self.Gbest - self.wolves[wolf] != 0:
                self.wolves[wolf] += self.step_x * ((self.Gbest - self.wolves[wolf])/np.linalg.norm(self.Gbest - self.wolves[wolf]))
                if self.minimize == True and self.func(self.wolves[wolf]) < self.Gbest:
                    self.Gbest = self.func(self.wolves[wolf])
                    self.x_wolf = self.wolves[wolf]
                elif self.minimize == False and self.func(self.wolves[wolf]) > self.Gbest:
                    self.Gbest = self.func(self.wolves[wolf])
                    self.x_wolf = self.wolves[wolf]
            else:
                pass
    
    def run(self):
        for _ in range(self.max_iter):
            self.iter()
    
    def plot(self):
        self.run()
        wolves = np.linspace(self.x_min, self.x_max, num = 350)
        fig, ax = plt.subplots()
        ys = []
        for i in wolves:
            ys.append(self.func(i))
        ax.plot(wolves, ys, linewidth=2.0)
        ax.scatter(self.x_wolf, self.Gbest, c = 'orchid')
        plt.show()

    def plot_dynamic(self):
        wolves = np.linspace(self.x_min, self.x_max, num = self.s)
        xs = np.linspace(self.x_min, self.x_max, num = 350)
        ys = []
        for i in xs:
            ys.append(self.func(i))
        for _ in range(self.max_iter):
            clear_output(wait=True)
            plt.plot(xs, ys, linewidth=2.0)
            self.iter()
            for i in self.wolves:
                plt.scatter(i, self.func(i), c = 'lightblue')
            plt.scatter(self.x_wolf, self.Gbest, c = 'orchid')
            plt.show()
    
class Wolves_optimization_x_y:
    def __init__(self, func, max_iter, s, step_x, x_min, x_max, step_y, y_min, y_max, minimize):
        self.func = func
        self.max_iter = max_iter
        self.s = s
        self.step_x = step_x
        self.x_min = x_min
        self.x_max = x_max
        self.step_y = step_y
        self.y_min = y_min
        self.y_max = y_max
        self.minimize = minimize
        self.wolves_x, self.wolves_y = self.create_first_pop()
        self.x_wolf, self.y_wolf = self.wolves_x[0][0], self.wolves_y[0][0]
        self.Gbest = self.func(self.x_wolf, self.y_wolf)

    def create_first_pop(self):
        stp_x = abs((self.x_min - self.x_max))/self.s
        stp_y = abs((self.y_min - self.y_max))/self.s
        wolves_x, wolves_y = np.arange(self.x_min, self.x_max, stp_x), np.arange(self.y_min, self.y_max, stp_y)
        wolves_x, wolves_y = np.meshgrid(wolves_x, wolves_y)
        return wolves_x, wolves_y
    
    def iter(self):
        for i in range(self.wolves_x.shape[0]-1):
            for j in range(self.wolves_x.shape[0]-1):
                if self.Gbest != self.wolves_x[i][j]:
                    self.wolves_x[i][j] += self.step_x * ((self.Gbest - self.wolves_x[i][j])/np.linalg.norm(self.Gbest - self.wolves_x[i][j]))
                else:
                    pass
                if self.Gbest != self.wolves_y[i][j]:
                    self.wolves_y[i][j] += self.step_y * ((self.Gbest - self.wolves_y[i][j])/np.linalg.norm(self.Gbest - self.wolves_y[i][j]))
                else:
                    pass
                if self.minimize == True and self.func(self.wolves_x[i][j], self.wolves_y[i][j]) < self.Gbest:
                    self.x_wolf = self.wolves_x[i][j]
                    self.y_wolf = self.wolves_y[i][j]
                    self.Gbest = self.func(self.x_wolf, self.y_wolf)
                elif self.minimize == False and self.func(self.wolves_x[i][j], self.wolves_y[i][j]) > self.Gbest:
                    self.x_wolf = self.wolves_x[i][j]
                    self.y_wolf = self.wolves_y[i][j]
                    self.Gbest = self.func(self.x_wolf, self.y_wolf)

    def run(self):
        for _ in range(self.max_iter):
            self.iter()
                    
    def plot_x_and_y(self):
        self.run()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.arange(self.x_min, self.x_max, 0.25), np.arange(self.y_min, self.y_max, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.25)
        ax.set_zlim(Z.min(), Z.max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.scatter(self.x_wolf, self.y_wolf, self.Gbest, c = 'orchid')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def plot_dynamic(self):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        X, Y = np.arange(self.x_min, self.x_max, 0.25), np.arange(self.y_min, self.y_max, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        def update_graph(num):
            self.iter()
            z = self.func(self.wolves_x, self.wolves_y)
            all_pop._offsets3d = (self.wolves_x[num], self.wolves_y[num], z[num])
            best._offsets3d = ([self.x_wolf], [self.y_wolf], [self.func(self.x_wolf, self.y_wolf)])
            title.set_text('Wolves optimization, iteration {}'.format(num+1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('Wolves optimization')
        all_pop = ax.scatter([], [], [], c = 'cornflowerblue', zorder = 1)
        best = ax.scatter([self.x_min], [self.y_min], [self.func(self.x_min, self.y_min)], c = 'orchid', zorder = 2)
        surf = ax.plot_surface(X, Y, Z, cmap = cm.viridis, linewidth=0, antialiased = False, alpha=0.25)
        ani = animation.FuncAnimation(fig, update_graph, self.max_iter, interval=60, blit=False, repeat=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()