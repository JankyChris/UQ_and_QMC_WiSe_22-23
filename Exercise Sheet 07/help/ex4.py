# Written by Vesa Kaarnioja, 2022
# Model solution to exercise 4 of week 5 using Python
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Import data to Python and convert from MATLAB-style indexing (1,2,3,...)
# to Python-style indexing (0,1,2,...).
mat = scipy.io.loadmat('week5.mat') # load the mat file
stiffness = mat['stiffness']
mass = mat['mass']
nodes = mat['nodes']
element = -1+mat['element'] # Python-style indexing
interior = -1+np.array(mat['interior'][0]) # Python-style indexing
ncoord = mat['ncoord'][0,0] # turn into a scalar
#nelem = mat['nelem'][0,0] # unused
#centers = mat['centers'] # unused

# Solve the eigenvalue problem.
evals,evecs = eigsh(stiffness,k=1,M=mass,which='SM')
coef = np.sqrt(np.transpose(evecs) @ mass @ evecs) # compute the L2 norm
evecs = -evecs/coef; # normalize the eigenfunction
sol = np.zeros(ncoord) # initialize the solution vector
sol[interior] = evecs.reshape((interior.size,)) # plug in the eigenfunction values solved in the interior of the domain

# Plot the results over the actual FE mesh (note the use of "element" in trisurf!)
fig=plt.figure(figsize=plt.figaspect(1.0))
ax=fig.add_subplot(1,1,1,projection='3d')
ax.plot_trisurf(nodes[:,0],nodes[:,1],sol,triangles=element,cmap=plt.cm.rainbow)
plt.show()
