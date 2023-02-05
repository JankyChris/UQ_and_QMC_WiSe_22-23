import numpy as np
from scipy import interpolate
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from FEM import FEMdata, UpdateStiffness

def findSol(level:int):
    # Find FE solution on a coarse mesh
    mass,grad,nodes,element,interior,centers,ncoord,nelem = FEMdata(level)
    A = UpdateStiffness(grad,np.ones(centers.shape[0]))
    f = lambda x: x[:, 0]
    rhs = mass[interior,:] @ f(nodes)
    sol = np.zeros(ncoord)
    tmp = spsolve(A[np.ix_(interior,interior)],rhs)
    sol[interior] = tmp
    return sol, nodes, mass

if __name__ == "__main__":
    reference_level = 5
    sol_ref, nodes_ref, mass_ref = findSol(reference_level)
    errors = np.zeros(4)

    for level in range(4):
        # Find FE solution on a coarse mesh
        sol_coarse, nodes_coarse, _ = findSol(level+1)
        # Interpolate the coarse FE results onto dense FE mesh
        sol_interp = interpolate.griddata((nodes_coarse[:,0],nodes_coarse[:,1]),sol_coarse,(nodes_ref[:,0],nodes_ref[:,1]))
        # Compute the L2 error between the two FE solutions
        errors[level] = (np.sqrt((sol_ref-sol_interp).T @ mass_ref @ (sol_ref-sol_interp)))

    for level, error in enumerate(errors):
        print(f"Error at level {level+1}: {error}.")

    plt.plot(range(1,5), errors, label="Finite Element Errors", marker="+", ls="", color="purple")
    m,b = np.polyfit(range(1,5), errors, 1)
    plt.plot(range(1,5), m*range(1,5)+b, label=f"m = {m}", linestyle="dashed", color="orange")
    plt.legend()
    plt.show()
