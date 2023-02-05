import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from FEM import FEMdata, UpdateStiffness

if __name__ == "__main__":
    # Load FEM data
    mass, grad, nodes, element, interior, centers, ncoord, nelem = FEMdata(5)
    random_field_decay = 2.0
    f = lambda x: x[:,0] # source term
    rhs = mass[interior,:] @ f(nodes)
    n_qmc_nodes = 2**15
    generating_vector = np.loadtxt('Exercise Sheet 11/data/offtheshelf2048.txt')
    ones = np.ones(interior.shape[0])
    S = [2**k for k in range(1,12)]
    quantity_of_interest = []

    def solve(i,n,a,z):
            qmcnode = np.mod(i*z/n,1)-1/2
            A = UpdateStiffness(grad,a(qmcnode))
            sol = spsolve(A[np.ix_(interior,interior)],rhs)
            return ones @ mass[np.ix_(interior,interior)] @ sol # quantity of interest

    for s in S:
        indices = np.arange(1,s+1).reshape((s,1))
        deterministic = indices**(-random_field_decay) * np.sin(np.pi * indices * centers[:,0]) * np.sin(np.pi * indices * centers[:,1])
        a = lambda y: 2 + y @ deterministic
        z = generating_vector[:s]

        # Solve 2^15 PDEs in parallel
        print(f"Solving {n_qmc_nodes} PDEs in parallel for s = {s}...")
        t = time.perf_counter()
        results = Parallel(n_jobs=-1, verbose=0)(delayed(solve)(i,n_qmc_nodes,a,z) for i in range(n_qmc_nodes))
        print(f"Finished in {time.perf_counter()-t:.2f} seconds.")
        # Compute the QMC average
        quantity_of_interest.append(np.mean(results))

    dim_truncation_errors=[]
    I_2048 = quantity_of_interest[-1]

    for I in quantity_of_interest[:-1]:
        dim_truncation_errors.append(np.abs(I - I_2048))

    S = np.array(S[:-1])

    plt.loglog(S, dim_truncation_errors, label="Dimension Truncation Errors", marker="+", ls="", color="purple")
    m, b = np.polyfit(np.log10(S), np.log10(dim_truncation_errors), 1)
    plt.loglog(S, S**m * 10**b, label=f"m = {m}", linestyle="dashed", color="orange")
    plt.legend()
    plt.show()