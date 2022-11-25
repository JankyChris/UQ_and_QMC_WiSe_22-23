import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def I(s):
    # returns the exact value of the integral Is
    return (2**s * np.cos(2*np.pi + s/2) * np.sin(1/2)**s)

def Q(s, n):
    # returns the Monte-Carlo estimation of the Integral Is
    # for dimension s and number of samples n
    def f(t):
        return (np.cos(2 * np.pi * np.sum(t)))

    t = np.random.uniform(low=0.0, high=1.0, size=(n,s))

    return (1/n * np.sum([f(i) for i in t]))

def error(I, Q):
    return np.abs(I-Q)

n_s = [10**i for i in range(1, 4)]
n_k = 20

for s in n_s:

    # For each dimension s, the Monte-Carlo approximation is done 10 times 
    # and the error is averaged out between these 10 passes in order to
    # minimize outliers. This way the actual convergence rate can be assessed
    # more reliably.

    err = np.zeros((10, n_k))
    for j in tqdm(range(10)):
        for k in range(n_k):
            n = 2**k

            q = Q(s, n)
            i = I(s)
            err[j, k] = error(i, q)
    
    err = err.mean(axis=0)

    plt.loglog([2**k for k in range(n_k)], 
               err, 
               label="s = {}".format(s))

plt.loglog([2**k for k in range(n_k)], 
           [2**(-k/2) for k in range(n_k)], 
           label="error = 1/sqrt(n)", 
           linestyle="dotted")
plt.legend()
plt.xlabel("n")
plt.ylabel("error")
plt.show()