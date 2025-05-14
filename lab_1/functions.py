import numpy as np
def f(x):
    return (x**3)*(3-x**5)*np.sin(10*np.pi*x)
def erkli(X, Y):
    return -20*np.e**(-0.2*np.sqrt(0.5*(X**2 + Y**2))) - np.e**(0.5*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + np.e + 20
def branin(x1, x2):
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    t = 1/(8*np.pi)
    return (x2 - b*x1**2 + c*x1 - 6)**2 + 10*(1-t)*np.cos(x1) + 10
def easom(x1, x2):
    return -np.cos(x1)*np.cos(x2)*np.e**(-(x1-np.pi)**2 - (x2-np.pi)**2)
def goldstein_price(x1, x2):
    return (1+(x1+x2+1)**2 * (19-14*x1+3*x1**2 - 14*x2 + 6*x1*x2 +3*x2**2))*(30+(2*x1 - 3*x2**2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
def six_hump_camel(x1, x2):
    return (4-2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (-4+4*x2**2)*x2**2