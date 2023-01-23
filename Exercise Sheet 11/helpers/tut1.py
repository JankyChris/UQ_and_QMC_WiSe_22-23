# Written by Vesa Kaarnioja, 2023
# Demonstration on how to estimate the FE error for two FE solutions computed using two different FE meshes

import numpy as np
import scipy.io
from scipy import sparse, interpolate
from scipy.sparse.linalg import spsolve

def FEMdata(level):
    mat = scipy.io.loadmat('FEM' + str(level) + '.mat')
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
    # Find FE solution on a dense mesh
    mass,grad,nodes,element,interior,centers,ncoord,nelem = FEMdata(5)
    A = UpdateStiffness(grad,np.ones(centers.shape[0]))
    f = lambda x: x[:,0]
    rhs = mass[interior,:] @ f(nodes)
    sol_dense = np.zeros(ncoord)
    tmp = spsolve(A[np.ix_(interior,interior)],rhs)
    sol_dense[interior] = tmp
    nodes_dense = nodes # store FE nodes of dense mesh for later use
    mass_dense = mass # store mass matrix of dense mesh for later use
    
    # Find FE solution on a coarse mesh
    mass,grad,nodes,element,interior,centers,ncoord,nelem = FEMdata(4)
    nodes_coarse = nodes
    A = UpdateStiffness(grad,np.ones(centers.shape[0]))
    rhs = mass[interior,:] @ f(nodes)
    sol_coarse = np.zeros(ncoord)
    tmp = spsolve(A[np.ix_(interior,interior)],rhs)
    sol_coarse[interior] = tmp
    
    # Interpolate the coarse FE results onto dense FE mesh
    sol_interp = interpolate.griddata((nodes_coarse[:,0],nodes_coarse[:,1]),sol_coarse,(nodes_dense[:,0],nodes_dense[:,1]))
    
    # Compute the L2 error between the two FE solutions
    print(np.sqrt((sol_dense-sol_interp).T @ mass_dense @ (sol_dense-sol_interp)))