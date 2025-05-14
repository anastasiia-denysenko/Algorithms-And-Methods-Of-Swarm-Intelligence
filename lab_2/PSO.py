import random
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.ticker import LinearLocator
from copy import deepcopy

class PSO(object):
    def __init__(self, func, num_iter, a1, a2, size_pop, v_min, v_max, x_min, x_max, y_min, y_max, minimize = True):
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
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.minimize = minimize
        self.curr_iter = 0
        self.pop_x, self.pop_y, self.pop_z = self.create_first_population()
        self.x_bests = np.full((self.size_pop, self.size_pop), self.x_min)
        self.y_bests = np.full((self.size_pop, self.size_pop), self.y_min)
        self.z_bests = np.full((self.size_pop, self.size_pop), self.func(self.x_min, self.y_min))
        self.x_ideal, self.y_ideal, self.z_ideal = self.x_min, self.y_min, self.func(self.x_min, self.y_min)
        self.vel_x, self.vel_y = self.create_velocity_vector()
        self.diff_x = abs(self.x_max) + abs(self.x_min)
        self.diff_y = abs(self.y_max) + abs(self.y_min)
    def put_in_boundry(self, i, j, x = True):
        if x == True: 
            if self.pop_x[-1][i][j] <= self.x_min or self.pop_x[-1][i][j] >= self.x_max:
                self.vel_x[i][j] *= -1
                self.pop_x[-1][i][j] =  ((self.pop_x[-1][i][j] - self.x_min) % self.diff_x + self.diff_x) % self.diff_x + self.x_min
        else:
            if self.pop_y[-1][i][j] <= self.y_min or self.pop_y[-1][i][j] >= self.y_max:
                self.vel_x[i][j] *= -1
                self.pop_y[-1][i][j] = ((self.pop_y[-1][i][j] - self.y_min) % self.diff_y + self.diff_y) % self.diff_y + self.y_min

    def in_boundry(self, sworm, x=True):
        if x == True:
            if sworm >= self.x_min and sworm <= self.x_max:
                return True
            else:
                False
        else:
            if sworm >= self.y_min and sworm <= self.y_max:
                return True
            else:
                return False
    def create_first_population(self):
        self.pop_x = []
        self.pop_y = []
        stp_x = abs((self.x_min - self.x_max))/self.size_pop
        stp_y = abs((self.y_min - self.y_max))/self.size_pop
        p_x, p_y = np.arange(self.x_min, self.x_max, stp_x), np.arange(self.y_min, self.y_max, stp_y)
        p_x, p_y = np.meshgrid(p_x, p_y)
        p_x, p_y = p_x.tolist(), p_y.tolist()
        self.pop_x.append(list(p_x))
        self.pop_y.append(list(p_y))
        self.pop_z = deepcopy(self.pop_x)
        for i in range(self.size_pop):
            for j in range(self.size_pop):
                self.pop_z[-1][i][j] = self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j])
        return self.pop_x, self.pop_y, self.pop_z
    def find_fitness(self):
        if self.minimize == True:
            for i in range(self.size_pop):
                for j in range(self.size_pop):
                    if self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j]) < self.func(self.x_bests[i][j], self.y_bests[i][j]):
                        self.x_bests[i][j] = self.pop_x[-1][i][j]
                        self.y_bests[i][j] = self.pop_y[-1][i][j]
                        self.z_bests[i][j] = self.func(self.x_bests[i][j], self.y_bests[i][j])
                        if self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j]) < self.z_ideal and self.in_boundry(self.pop_x[-1][i][j], True) == True and self.in_boundry(self.pop_y[-1][i][j], False) == True:
                            self.x_ideal = self.pop_x[-1][i][j]
                            self.y_ideal = self.pop_y[-1][i][j]
                            self.z_ideal = self.func(self.x_ideal, self.y_ideal)
        else:
            for i in range(self.size_pop):
                for j in range(self.size_pop):
                    if self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j]) > self.func(self.x_bests[i][j], self.y_bests[i][j]):
                        self.x_bests[i][j] = self.pop_x[-1][i][j]
                        self.y_bests[i][j] = self.pop_y[-1][i][j]
                        self.z_bests[i][j] = self.func(self.x_bests[i][j], self.y_bests[i][j])
                        if self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j]) > self.z_ideal and self.pop_x[-1][i][j] >= self.x_min and self.pop_x[-1][i][j] <= self.x_max and self.pop_y[-1][i][j] >= self.y_min and self.pop_y[-1][i][j] <= self.y_max:
                            self.x_ideal = self.pop_x[-1][i][j]
                            self.y_ideal = self.pop_y[-1][i][j]
                            self.z_ideal = self.func(self.x_ideal, self.y_ideal)
        return self.x_bests, self.y_bests, self.z_bests, self.x_ideal, self.y_ideal, self.z_ideal
    def create_velocity_vector(self):
        if self.curr_iter == 0:
            self.vel_x, self.vel_y = [], []
            for i in range(self.size_pop):
                tmp_x = []
                tmp_y = []
                for j in range(self.size_pop):
                    tmp_x.append(self.v_min + (self.v_max - self.v_min)*random.uniform(0, 1))
                    tmp_y.append(self.v_min + (self.v_max - self.v_min)*random.uniform(0, 1))
                self.vel_x.append(tmp_x)
                self.vel_y.append(tmp_y)
        else:
            r1 = np.random.rand()
            r2 = np.random.rand()
            for i in range(self.size_pop):
                for j in range(self.size_pop):
                    self.vel_x[i][j] += self.a1*(self.x_bests[i][j]-self.pop_x[-1][i][j])*r1 + self.a2*(self.x_ideal - self.pop_x[-1][i][j])*r2
                    self.vel_y[i][j] += self.a1*(self.y_bests[i][j]-self.pop_y[-1][i][j])*r1 + self.a2*(self.y_ideal - self.pop_y[-1][i][j])*r2
        return self.vel_x, self.vel_y
    def update_positions(self):
        self.pop_x.append(self.pop_x[-1])
        self.pop_y.append(self.pop_y[-1])
        self.pop_z.append(self.pop_z[-1])
        for i in range(self.size_pop):
             for j in range(self.size_pop):
                self.pop_x[-1][i][j] += self.vel_x[i][j]
                self.pop_y[-1][i][j] += self.vel_y[i][j]
                self.put_in_boundry(i, j, True)
                self.put_in_boundry(i, j, False)
                self.pop_z[-1][i][j] = self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j])
        return self.pop_x[-1], self.pop_y[-1]
    def iter(self):
        self.find_fitness()
        self.create_velocity_vector()
        self.update_positions()
        self.curr_iter += 1
    def run(self):
        while self.curr_iter != self.num_iter:
            self.iter()
        return self.pop_x[-1], self.pop_y[-1], self.x_ideal, self.y_ideal, self.z_ideal
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
        ax.scatter(self.x_ideal, self.y_ideal, self.z_ideal, c = 'orchid')
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
                all_pop._offsets3d = (self.pop_x[-1][i], self.pop_y[-1][i], self.pop_z[-1][i])
                best._offsets3d = ([self.x_ideal], [self.y_ideal], [self.func(self.x_ideal, self.y_ideal)])
            title.set_text('PSO, iteration {}'.format(num+1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('PSO')
        all_pop = ax.scatter(np.concatenate(self.pop_x[0]), np.concatenate(self.pop_y[0]), np.concatenate(self.pop_z[0]), c = 'cornflowerblue', zorder = 1)
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