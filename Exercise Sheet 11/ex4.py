"""
@Author:    Christoph M. Jankowsky
@Date:      2023-02-05

My solutions to Sheet 11 - Exercise 4 of the course 
Uncertainty Quantification and Quasi-Monte Carlo at FU Berlin.
"""

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm # need inverse CDF for normal distribution
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from FEM import FEMdata, UpdateStiffness

if __name__ == "__main__":
    # Load FEM data
    mass, grad, nodes, element, interior, centers, ncoord, nelem = FEMdata(5)

    random_field_decay = 2.0
    s = 100
    indices = np.arange(1,s+1).reshape((s,1))
    deterministic = indices**(-random_field_decay) * np.sin(np.pi * indices * centers[:,0]) * np.sin(np.pi * indices * centers[:,1])
    a = lambda y: np.exp(y @ deterministic)

    f = lambda x: x[:,0] # source term
    rhs = mass[interior,:] @ f(nodes)

    R = 16
    generating_vector = np.loadtxt('Exercise Sheet 11/data/offtheshelf2048.txt')
    z = generating_vector[:s]

    ones = np.ones(interior.shape[0])

    def solve(i,shift,n):
        qmcnode = norm.ppf(np.mod(i*z/n+shift,1)) # use inverse CDF here!
        A = UpdateStiffness(grad,a(qmcnode))
        sol = spsolve(A[np.ix_(interior,interior)],rhs)
        return ones @ mass[np.ix_(interior,interior)] @ sol

    errors = []
    N = [2**k for k in range(10, 16)]

    for n_qmc_nodes in N:
        results = []
    
        # For each random shift, compute the QMC estimate using n cubature nodes
        print(f"Solving {n_qmc_nodes} PDEs in parallel with {R} random shifts...")
        t = time.perf_counter()
        with Parallel(n_jobs=-1, verbose=0) as parallel:
            for r in range(R):
                shift = np.random.uniform(0,1,s)
                tmp = parallel(delayed(solve)(i,shift,n_qmc_nodes) for i in range(n_qmc_nodes))
                results.append(np.mean(tmp))
        print(f"Finished in {time.perf_counter()-t:.2f} seconds.")
            
        # Take the average of the R QMC approximations as the final estimate
        qmcavg = np.mean(results)
        rmserror = np.linalg.norm(qmcavg-results)/np.sqrt(R*(R-1))
        
        errors.append(rmserror)

    plt.loglog(N, errors, label="QMC Errors (lognormal)", marker="+", ls="", color="purple")
    m, b = np.polyfit(np.log10(N), np.log10(errors), 1)
    plt.loglog(N, N**m * 10**b, label=f"m = {m}", linestyle="dashed", color="orange")
    plt.legend()
    plt.show()
    