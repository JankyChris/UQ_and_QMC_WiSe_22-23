# Written by Christoph Jankowsky - 2022
# Exercise 7.1 of the course Uncertainty Quantification and Quasi-Monte Carlo

import numpy as np
import scipy.io
from scipy import sparse

def FEMdata():
    """
    Load FEM matrices from "week7.mat" file and convert MATLAB-specifics
    for use with python3.
    """
    mat = scipy.io.loadmat("week7.mat")

    grad = mat["grad"]
    mass = mat["mass"]
    nodes = mat["nodes"]
    element = -1+mat['element']                 # Python-style indexing
    interior = -1+np.array(mat['interior'][0])  # Python-style indexing
    centers = mat["centers"]
    ncoord = mat['ncoord'][0,0]                 # turn into a scalar
    nelem = mat['nelem'][0,0]                   # turn into a scalar


    return grad, mass, nodes, element, interior, centers, ncoord, nelem

f = lambda x: x[0]

def a(x, y):
    """
    Definition of the diffusion coefficient.
    """

    def psi(k, x):
        """
        Stochastic fluctuations.
        """
        x1 = x[0]
        x2 = x[1]

        return k**(-2) * np.sin(pi*k*x1) * np.sin(pi*k*x2)
    
    kmax = len(y)
    val = []

    for k in range(kmax):
        val.append(y[k] * psi(k, x))

    return 2 + np.sum(val)

def UpdateStiffness(grad, a):
    n = np.sqrt(grad.shape[0]).astype(int)
    return (grad@sparse.csr_matrix(a.reshape((a.size,1)))).reshape((n,n))

if __name__ == "__main__":
    grad, mass, nodes, element, interior, centers, ncoord, nelem = FEMdata()
    a = ...
    stiff = UpdateStiffness(grad, a)
