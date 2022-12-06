# Written by Christoph Jankowsky - 2022
# Exercise 7.1 of the course Uncertainty Quantification and Quasi-Monte Carlo

import numpy as np
import scipy.io
from scipy import sparse
from scipy.sparse.linalg import eigsh

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

def a(centers, y):
    """
    Definition of the diffusion coefficient.
    """

    return k**(-2) * np.sin(np.pi*k*centers[:,0]) * np.sin(np.pi*k*centers[:,1])
    
    """
    kmax = len(y)
    val = []

    for k in range(kmax):
        val.append(y[k] * psi(k, x))

    return 2 + np.sum(val)
    """

def UpdateStiffness(grad, a):
    n = np.sqrt(grad.shape[0]).astype(int)
    return (grad@sparse.csr_matrix(a.reshape((a.size,1)))).reshape((n,n))

def MonteCarlo(y, x, u, n, s=100):
    us = [u(x, yi) for yi in y]
    
    return (1/n * np.sum(us))

if __name__ == "__main__":
    # use n == 2^maxiter as the "reference solution"
    maxiter = 20;

    # load FEM data from MATLAB file
    grad, mass, nodes, element, interior, centers, ncoord, nelem = FEMdata()

    # define source term ( f(x)=x_1 )
    f = lambda x: x[:, 0]
    # define loading vector
    rhs = mass[interior, :] @ f(nodes)

    sums = []
    means = []
    null = np.zeros(ncoord, 1)

    # range over an increasing number of random samples
    for i in range(maxiter):
        print("Iteration: ", i)
        n = max(1, 2**(i-1))
        tmp = np.zeros(ncoord, 1)
        # main loop
        for k in range(1, n+1):
            mcnode = np.random.uniform(low=-0.5, high=0.5, size=(n,s))
            A = UpdateStiffness(grad, a(centers, mcnode))
            sol = null
            sol[interior] = np.linalg.solve(A[interior, interior], rhs)