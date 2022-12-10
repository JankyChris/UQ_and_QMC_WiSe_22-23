# Written by Christoph Jankowsky - 2022
# Exercise 7 of the course Uncertainty Quantification and Quasi-Monte Carlo
# using a classic Monte Carlo method

import numpy as np
import scipy.io
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
from time import perf_counter
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def FEMdata(path="Exercise Sheet 07/week7.mat"):
    """
    Load FEM matrices from "week7.mat" file and convert MATLAB-specifics
    for use with python3.
    """
    mat = scipy.io.loadmat(path)

    grad = mat["grad"]                      # gradient part of stiffness matrix/tensor
    mass = mat["mass"]                      # mass matrix
    nodes = mat["nodes"]                    # FE nodes
    element = -1 + mat['element']           # indices for the elements (Python-style indexing)
    interior = -1 + mat['interior'][:, 0]   # indices for the interior nodes (Python-style indexing)
    centers = mat["centers"]                # coordinates of the element centers
    ncoord = mat['ncoord'][0,0]             # number of FE nodes (to scalar)
    nelem = mat['nelem'][0,0]               # Number of elements (to scalar)

    return grad, mass, nodes, element, interior, centers, ncoord, nelem

def a(centers, y, s=100, decay=2.0):
    """
    Definition of the diffusion coefficient.
    """
    indices = np.arange(1, s+1).reshape((s, 1))
    deterministic = indices**(-decay) * np.sin(np.pi * indices * centers[:,0]) * np.sin(np.pi * indices * centers[:,1])

    return 2 + y @ deterministic

def UpdateStiffness(grad, a):
    """
    Generates the stiffness matrix from given grad and diffusion coefficient a
    """
    n = np.sqrt(grad.shape[0]).astype(int)
    vec = grad @ sparse.csr_matrix(a.reshape((a.size, 1)))

    return sparse.csr_matrix.reshape(vec,(n,n)).tocsr()

def mse(A, B):
    """
    Returns the mean-squared (L2) error of two arrays A and B
    """
    return ((A-B)**2).mean()

def plotPDE(nodes, element, result):
    """
    Plots the PDE solution
    """
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1,1,1,projection="3d")
    ax.plot_trisurf(nodes[:, 0], nodes[:, 1], result, triangles=element, cmap="viridis")
    plt.show()

def plotERROR(reference, approximations, n):
    """
    Plots the log-log error graph between the reference solution and the array of approximations
    """
    ns = np.asarray([i for i in range(int(n/(2**3)))])
    errors = [mse(reference, approximation) for approximation in approximations]
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1,1,1)
    ax.loglog(ns, errors, label=...)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # use n == 2^maxiter as the "reference solution"
    maxiter = 18;

    # set stochastic dimension
    s = 100

    # load FEM data from MATLAB file
    grad, mass, nodes, element, interior, centers, ncoord, nelem = FEMdata()

    # set up the loading term
    f = lambda x: x[:, 0]               # define source term ( f(x)=x_1 )
    rhs = mass[interior, :] @ f(nodes)  # define loading vector

    # Solve the PDE n = 2^maxiter times
    n = 2**maxiter

    ## Source problem using Monte Carlo method
    """
    # Sanity check: Can we solve the PDE for one realization of the uniform random vector y?
    print('Solve one PDE source problem...')
    start = perf_counter()
    mcnode = np.random.uniform(-1/2,1/2,s)
    A = UpdateStiffness(grad,a(centers, mcnode, s))
    sol = np.zeros(ncoord)
    tmp = spsolve(A[np.ix_(interior,interior)], rhs)
    sol[interior] = tmp
    stop = perf_counter()
    print(f"Solving took {stop-start:.2f} seconds.")

    # Visualize the results
    plotPDE(nodes, element, sol)
    """

    n_cpu = 16

    def solveMC(i):
        mcnode = np.random.uniform(-1/2,1/2,s)
        A = UpdateStiffness(grad,a(centers, mcnode, s))
        sol = np.zeros(ncoord)
        tmp = spsolve(A[np.ix_(interior,interior)], rhs)
        sol[interior] = tmp
        return sol

    print(f"Solve {n:,} PDE source problems in parallel with {n_cpu} workers using Monte Carlo nodes...")
    start = perf_counter()
    resultsMC = np.asarray(Parallel(n_jobs=n_cpu, verbose=0)(delayed(solveMC)(i) for i in range(n)))
    stop = perf_counter()
    print(f"Solving took {stop-start:.2f} seconds.")

    referenceMC = resultsMC.mean(axis=0)

    # Visualize the results
    #plotPDE(nodes, element, reference)
    approximationMC = [resultsMC[:i+1].mean(axis=0) for i in range(int(n/(2**3)))]

    ## Source Problem using Quasi-Monte Carlo method

    z = np.loadtxt('Exercise Sheet 07/offtheshelf.txt') # Generating vector

    def solveQMC(i):
        qmcnode = np.mod(i*z/n,1)-1/2
        A = UpdateStiffness(grad, a(centers, qmcnode, s))
        sol = np.zeros(ncoord)
        tmp = spsolve(A[np.ix_(interior, interior)], rhs)
        sol[interior] = tmp
        return sol

    print(f"Solve {n:,} PDE source problems in parallel with {n_cpu} workers using Quasi-Monte Carlo nodes...")
    start = perf_counter()
    resultsQMC = np.asarray(Parallel(n_jobs=n_cpu)(delayed(solveQMC)(i) for i in range(n)))
    stop = perf_counter()
    print(f"Solving took {stop-start:.2f} seconds.")

    referenceQMC = resultsQMC.mean(axis=0)

    # Visualize the results
    #plotPDE(nodes, element, reference)
    approximationQMC = [resultsQMC[:i+1].mean(axis=0) for i in range(int(n/(2**3)))]

    plotPDE(nodes, element, referenceQMC)

    plotERROR(referenceQMC, approximationQMC, n)