import numpy as np
def mishra_bird(x, y = None):
    if (isinstance(x, np.ndarray) or type(x) == list) and y is not None:
        res = np.full((len(x), len(x[0])), 0.0)
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = np.sin(y[i][j]) * np.exp((1 - np.cos(x[i][j]))**2) + np.cos(y[i][j]) + np.cos(x[i][j]) * np.exp((1 - np.sin(x[i][j]))**2) + (x[i][j] - y[i][j])**2
        return res
    elif (isinstance(x, np.ndarray) or type(x) == list) and y is None:
        x, y = x
        return np.sin(y) * np.exp((1 - np.cos(x))**2) + np.cos(y) + np.cos(x) * np.exp((1 - np.sin(x))**2) + (x - y)**2
    else:
        return np.sin(y) * np.exp((1 - np.cos(x))**2) + np.cos(y) + np.cos(x) * np.exp((1 - np.sin(x))**2) + (x - y)**2

def rastrigin(x, A = 10):
    x = np.array(x)
    n = x.size
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def Rosenbrock(x, y = None):
    if (isinstance(x, np.ndarray) or type(x) == list) and y is not None:
        res = np.full((len(x), len(x[0])), 0.0)
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = (1 - x[i][j])**2 + 100*(y[i][j] - x[i][j]**2)**2 if x[i][j]**2+y[i][j]**2 < 2 else np.inf
        return res
    elif (isinstance(x, np.ndarray) or type(x) == list) and y is None:
        x, y = x
        if x**2+y**2 < 2:
            return (1 - x)**2 + 100*(y - x**2)**2
        else:
            return float('inf')
    else:
        if x**2+y**2 < 2:
            return (1 - x)**2 + 100*(y - x**2)**2
        else:
            return float('inf')
    
def Sumonesku(X, Y = None):
    if isinstance(X, np.ndarray) or type(X) == list and Y is not None:
        res = np.full((len(X), len(X[0])), 0.0)
        for i in range(len(X)):
            for j in range(len(X[0])):
                res[i][j] = 0.1*X*Y if X**2 + Y**2 < (1+0.2*np.cos(8*np.arctan(X/Y)))**2 else float('inf')
        return res
    elif isinstance(X, np.ndarray) or type(X) == list and Y is None:
        x, y = X
        if x**2 + y**2 < (1+0.2*np.cos(8*np.arctan(x/y)))**2:
            return 0.1*x*y
        else:
            return float('inf')
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