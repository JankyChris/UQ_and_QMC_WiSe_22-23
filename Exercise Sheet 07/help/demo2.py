# Written by Vesa Kaarnioja, 2022
# Demonstration on how to solve parametric PDE source and eigenvalue
# problems over an off-the-shelf quasi-Monte Carlo point set
import numpy as np
import math
import scipy.io
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def UpdateStiffness(grad,a):
    n = np.sqrt(grad.shape[0]).astype(int)
    vec = grad@sparse.csr_matrix(a.reshape((a.size,1)))
    return sparse.csr_matrix.reshape(vec,(n,n)).tocsr()

if __name__ == '__main__':
    # Import FEM data
    mat = scipy.io.loadmat('Exercise Sheet 07/week7.mat')
    grad = mat['grad']
    mass = mat['mass']
    nodes = mat['nodes']
    element = -1 + mat['element']
    interior = -1 + mat['interior'][:,0]
    ncoord = mat['ncoord'][0,0]
    nelem = mat['nelem'][0,0]
    centers = mat['centers']

    # Set up the diffusion coefficient
    s = 100 # stochastic dimension
    decay = 2.0 # decay of input random field
    indices = np.arange(1,s+1).reshape((s,1))
    deterministic = indices**(-decay) * np.sin(math.pi * indices * centers[:,0]) * np.sin(math.pi * indices * centers[:,1])
    a = lambda y: 2 + y @ deterministic

    # Set up the loading
    f = lambda x: x[:,0] # source term
    rhs = mass[interior,:] @ f(nodes) # loading vector
    
    z = np.loadtxt('Exercise Sheet 07/offtheshelf.txt') # Generating vector
    
    ## Part 1: Source problem
    # Solve PDE for many realization of a random vector
    n = 2**10
    print('Solve many PDE source problems')
    def solve(i):
        qmcnode = np.mod(i*z/n,1)-1/2
        A = UpdateStiffness(grad,a(qmcnode))
        sol = np.zeros(ncoord)
        tmp = spsolve(A[np.ix_(interior,interior)],rhs)
        sol[interior] = tmp
        return sol
        
    results = Parallel(n_jobs=10)(delayed(solve)(i) for i in range(n))
    
    # Visualize the results
    for i in [0,59,1023]:
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.plot_trisurf(nodes[:,0],nodes[:,1],results[i],triangles=element,cmap="viridis")
        plt.show()

    ## Part 2: Eigenvalue problem
    # Sanity check: Can we solve the eigenpair for one realization of the uniform random vector y?

    # Solve the PDE for many realizations of the uniform random vector
    n = 2**10
    print('Solve many PDE eigenvalue problems')
    def solve(i):
        qmcnode = np.mod(i*z/n,1)-1/2
        stiffness = UpdateStiffness(grad,a(qmcnode))
        evals,evecs = eigsh(stiffness[np.ix_(interior,interior)],k=1,M=mass[np.ix_(interior,interior)],which='SM')
        coef = np.sqrt(evecs.T @ mass[np.ix_(interior,interior)] @ evecs)
        evecs = np.sign(evecs[0])*evecs/coef
        evecs_full = np.zeros(ncoord)
        evecs_full[interior] = evecs.T
        evals = evals[0]
        return evals,evecs_full
    
    results = Parallel(n_jobs=10)(delayed(solve)(i) for i in range(n))
    for i in [0,44,1023]:
        print(str(i+1) + 'th eigenvalue: ' + str(results[i][0]))
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.plot_trisurf(nodes[:,0],nodes[:,1],results[i][1],triangles=element,cmap="viridis")
        plt.show()
