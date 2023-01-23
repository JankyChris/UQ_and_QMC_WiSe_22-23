# Written by Vesa Kaarnioja, 2023
# Demonstration on how to estimate the R.M.S. error for a 100-dimensional integral in the lognormal setting with 4 random shifts and a _single_ fixed value for the number of QMC nodes n

import numpy as np
import scipy.io
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
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
    mass,grad,nodes,element,interior,centers,ncoord,nelem = FEMdata(3)
    
    # Set up the diffusion coefficient
    s = 100 # stochastic dimension
    decay = 2.0 # decay of the input random field
    indices = np.arange(1,s+1).reshape((s,1))
    deterministic = indices**(-decay) * np.sin(np.pi * indices * centers[:,0]) * np.sin(np.pi * indices * centers[:,1])
    a = lambda y: np.exp(y @ deterministic)
    
    # Set up the loading
    f = lambda x: x[:,0]
    rhs = mass[interior,:] @ f(nodes)

    # Import the generating vector
    gen = np.loadtxt('Exercise Sheet 11/data/offtheshelf2048.txt')
    z = gen[:s] # truncate the generating vector to s terms
    n = 2**15 # number of QMC nodes
    R = 4 # number of random shifts
    ones = np.ones(interior.shape[0])
    
    def solve(i,shift,n):
        qmcnode = norm.ppf(np.mod(i*z/n+shift,1)) # use inverse CDF here!
        A = UpdateStiffness(grad,a(qmcnode))
        sol = spsolve(A[np.ix_(interior,interior)],rhs)
        return ones @ mass[np.ix_(interior,interior)] @ sol
    
    results = []
    
    # For each random shift, compute the QMC estimate using n cubature nodes
    with Parallel(n_jobs=-1, verbose=1) as parallel:
        for r in range(R):
            shift = np.random.uniform(0,1,s)
            tmp = parallel(delayed(solve)(i,shift,n) for i in range(n))
            results.append(np.mean(tmp))
            
    # Take the average of the R QMC approximations as the final estimate
    qmcavg = np.mean(results)
    rmserror = np.linalg.norm(qmcavg-results)/np.sqrt(R*(R-1)) # R.M.S. error estimate
    print(rmserror)