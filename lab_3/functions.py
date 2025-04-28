import numpy as np

def mishra_bird(arg):
    x, y = arg
    if isinstance(x, np.ndarray) or type(x) == list:
        if x.ndim == 2:
            res = np.full((len(x), len(x[0])), 0.0)
            for i in range(len(x)):
                for j in range(len(x[0])):
                    res[i][j] = np.sin(y[i][j]) * np.exp((1 - np.cos(x[i][j]))**2) + np.cos(y[i][j]) + np.cos(x[i][j]) * np.exp((1 - np.sin(x[i][j]))**2) + (x[i][j] - y[i][j])**2
        elif x.ndim == 1:
            res = [0]*len(x)
            for i in range(len(x)):
                res[i] = np.sin(y[i]) * np.exp((1 - np.cos(x[i]))**2) + np.cos(y[i]) + np.cos(x[i]) * np.exp((1 - np.sin(x[i]))**2) + (x[i] - y[i])**2
    else:
        res = np.sin(y) * np.exp((1 - np.cos(x))**2) + np.cos(y) + np.cos(x) * np.exp((1 - np.sin(x))**2) + (x - y)**2
    return res

def rastring(x):
    if isinstance(x, int) or isinstance(x, float):
        x = [x]
    n = len(x)
    x = np.array([np.array(xs, dtype=np.float64) for xs in x])
    return 10*n + sum([np.power(xi, 2) - 10*np.cos(np.multiply(np.pi, np.multiply(2, xi))) for xi in x])
    
a, b = 1, 3
n = 1000
step = (b-a)/n
t = np.linspace(a, b, n)
alpha = np.linspace(-3, 3, 100)
x, y, z = [0]*len(t), [0]*len(t), [0]*len(t)
x[0] = 2
y[0] = 1
y_square = []
y_abs = []
for a in alpha:
    z[0] = a
    for i in range(t.shape[0]-1):
        x[i+1] = (2*x[i-1]**2 - 25*t[i]**2 - np.sin(x[i]*y[i]*t[i]))*step + x[i]
        y[i+1] = (1 - 4*np.cos(x[i+1]*t[i]))*step + z[i]
        z[i+1] = z[i]*step + y[i]
    y_abs.append(abs(y[-1] + 1))
    y_square.append((y[-1] + 1)**2)

yabsolute = y_abs[:]
ysquared = y_square[:]

def linear_interpolation2d(alpha, y, x):
    n = len(alpha)
    for i in range(1, n):
        if alpha[i] > x:
            return (x - alpha[i - 1]) / (alpha[i] - alpha[i - 1]) * (y[i] - y[i - 1]) + y[i - 1]
    if x <= alpha[0]:
        return y[0]
    elif x >= alpha[-1]:
        return y[-1]
    return None

def absolute(x):
    if type(x) == list:
         x = x[0]
    if isinstance(x, np.ndarray) or type(x) == list:
        if x.ndim == 2:
            l = np.full((len(x), len(x[0])), 0.0)
            for i in range(len(x)):
                for j in range(len(x[0])):
                    l[i][j] = linear_interpolation2d(alpha, y_abs, x[i][j])
                    if l[i][j] == None:
                        raise ValueError("None value in interpolation output")
            return l
        elif x.ndim == 1:
            l = [0]*len(x)
            for i in range(len(x)):
                l[i] = linear_interpolation2d(alpha, y_abs, x[i])
            return l
    else:
        return linear_interpolation2d(alpha, y_abs, x)

def squared(x):
    if type(x) == list:
         x = x[0]
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            l = np.full((len(x), len(x[0])), 0.0)
            for i in range(len(x)):
                for j in range(len(x[0])):
                    l[i][j] = linear_interpolation2d(alpha, y_square, x[i][j])
                    if l[i][j] == None:
                        raise ValueError("None value in interpolation output")
            return l
        elif x.ndim == 1:
            l = [0]*len(x)
            for i in range(len(x)):
                l[i] = linear_interpolation2d(alpha, y_square, x[i])
            return l
    else:
        return linear_interpolation2d(alpha, y_square, x)