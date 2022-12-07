# Written by Christoph Jankowsky - 2022
# Exercise 7.1 of the course Uncertainty Quantification and Quasi-Monte Carlo

import numpy as np
import scipy.io
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
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


# TODO: Monte Carlo sampler

def MonteCarlo():
    ...

def plotPDE(nodes, element, result):
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1,1,1,projection="3d")
    ax.plot_trisurf(nodes[:, 0], nodes[:, 1], result, triangles=element, cmap="viridis")
    plt.show()

if __name__ == "__main__":
    # use n == 2^maxiter as the "reference solution"
    maxiter = 20;

    # set stochastic dimension
    s = 100

    # load FEM data from MATLAB file
    grad, mass, nodes, element, interior, centers, ncoord, nelem = FEMdata()

    # set up the loading term
    f = lambda x: x[:, 0]               # define source term ( f(x)=x_1 )
    rhs = mass[interior, :] @ f(nodes)  # define loading vector

    ## Part 1: Source problem
    # Sanity check: Can we solve the PDE for one realization of the uniform random vector y?
    print('Solve one PDE source problem')
    mcnode = np.random.uniform(-1/2,1/2,s)
    A = UpdateStiffness(grad,a(centers, mcnode, s))
    sol = np.zeros(ncoord)
    tmp = spsolve(A[np.ix_(interior,interior)], rhs)
    sol[interior] = tmp

    # Visualize the results
    plotPDE(nodes, element, sol)
