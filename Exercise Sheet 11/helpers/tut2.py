# Written by Vesa Kaarnioja, 2023
# Demonstration on how to compute a 2048-dimensional reference solution, to be used for dimension truncation error estimation

import numpy as np
import scipy.io
from scipy import sparse
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed

def FEMdata(level):
    mat = scipy.io.loadmat('Exercise Sheet 11/data/FEM' + str(level) + '.mat')
    grad = mat['grad']
    mass = mat['mass']
    nodes = mat['nodes']
    element = -1 + mat['element']
    interior = -1 + mat['interior'][:,0]
    centers = mat['centers']
    ncoord = mat['ncoord'][0,0]
    nelem = mat['nelem'][0,0]
    return mass,grad,nodes,element,interior,centers,ncoord,nelem

def UpdateStiffness(grad,a):
    n = np.sqrt(grad.shape[0]).astype(int)
    vec = grad @ sparse.csr_matrix(a.reshape((a.size,1)))
    return sparse.csr_matrix.reshape(vec,(n,n)).tocsr()

if __name__ == '__main__':
    # Load FEM data
    mass,grad,nodes,element,interior,centers,ncoord,nelem = FEMdata(5)
    
    # Set up the diffusion coefficient
    s = 2048 # stochastic dimension
    decay = 2.0 # decay of the input random field
    indices = np.arange(1,s+1).reshape((s,1))
    deterministic = indices**(-decay) * np.sin(np.pi * indices * centers[:,0]) * np.sin(np.pi * indices * centers[:,1])
    a = lambda y: 2 + y @ deterministic
    
    # Set up the loading
    f = lambda x: x[:,0]
    rhs = mass[interior,:] @ f(nodes)

    # Import the generating vector
    gen = np.loadtxt('Exercise Sheet 11/data/offtheshelf2048.txt')
    z = gen[:s] # truncate generating vector to s terms
    n = 2**15 # fix number of QMC nodes
    ones = np.ones(interior.shape[0])
    
    def solve(i,n):
        qmcnode = np.mod(i*z/n,1)-1/2
        A = UpdateStiffness(grad,a(qmcnode))
        sol = spsolve(A[np.ix_(interior,interior)],rhs)
        return ones @ mass[np.ix_(interior,interior)] @ sol # quantity of interest
    
    # Solve 2^15 PDEs in parallel
    results = Parallel(n_jobs=-1, verbose=5)(delayed(solve)(i,n) for i in range(n))
    
    # Compute the QMC average
    reference = np.mean(results)
    print(reference)

