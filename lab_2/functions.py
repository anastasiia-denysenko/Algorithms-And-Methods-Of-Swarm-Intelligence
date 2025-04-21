import numpy as np
def mishra_bird(x, y):
    if isinstance(x, np.ndarray) or type(x) == list:
        res = np.full((len(x), len(x[0])), 0.0)
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = np.sin(y[i][j]) * np.exp((1 - np.cos(x[i][j]))**2) + np.cos(y[i][j]) + np.cos(x[i][j]) * np.exp((1 - np.sin(x[i][j]))**2) + (x[i][j] - y[i][j])**2
        return res
    else:
        return np.sin(y) * np.exp((1 - np.cos(x))**2) + np.cos(y) + np.cos(x) * np.exp((1 - np.sin(x))**2) + (x - y)**2

def rastrigin(*X, **kwargs):
    A = kwargs.get('A', 10)
    return A + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

def Rosenbrock(x, y):
    if isinstance(x, np.ndarray) or type(x) == list:
        res = np.full((len(x), len(x[0])), 0.0)
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = (1 - x[i][j])**2 + 100*(y[i][j] - x[i][j]**2)**2 if x[i][j]**2+y[i][j]**2 < 2 else np.inf
        return res
    else:
        if x**2+y**2 < 2:
            return (1 - x)**2 + 100*(y - x**2)**2
        else:
            return float('inf')
    
def Sumonesku(X, Y):
    if isinstance(X, np.ndarray) or type(X) == list:
        res = np.full((len(X), len(X[0])), 0.0)
        for i in range(len(X)):
            for j in range(len(X[0])):
                res[i][j] = 0.1*X*Y if X**2 + Y**2 < (1+0.2*np.cos(8*np.arctan(X/Y)))**2 else float('inf')
        return res
    else:
        if X**2 + Y**2 < (1+0.2*np.cos(8*np.arctan(X/Y)))**2:
            return 0.1*X*Y
        else:
            return float('inf')
    
def erkli(X, Y):
    if isinstance(X, np.ndarray) or type(X) == list:
        res = np.full((len(X), len(X[0])), 0.0)
        for i in range(len(X)):
            for j in range(len(X[0])):
                res[i][j] = -20*np.e**(-0.2*np.sqrt(0.5*(X[i][j]**2 + Y[i][j]**2))) - np.e**(0.5*(np.cos(2*np.pi*X[i][j]) + np.cos(2*np.pi*Y[i][j]))) + np.e + 20
        return res
    else:
        return -20*np.e**(-0.2*np.sqrt(0.5*(X**2 + Y**2))) - np.e**(0.5*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + np.e + 20