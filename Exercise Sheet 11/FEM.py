import numpy as np
import scipy.io
from scipy import sparse

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