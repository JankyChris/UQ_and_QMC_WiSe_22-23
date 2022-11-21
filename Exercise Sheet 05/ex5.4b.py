# Solution to Exercise 5.3 of the course
# Uncertainty Quantification and Quasi-Monte Carlo
# by Christoph Jankowsky

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.sparse.linalg import eigsh
from scipy.interpolate import griddata

mat = scipy.io.loadmat("Exercise Sheet 05/data/week5.mat")
mass = mat["mass"]
stiffness = mat["stiffness"]
nodes = mat["nodes"]
element = mat["element"]
interior = mat["interior"]
centers = mat["centers"]
ncoord = mat["ncoord"]
nelem = mat["nelem"]

x = [nodes[n-1, 0] for n in interior]
y = [nodes[n-1, 1] for n in interior]

evals, evecs = eigsh(stiffness, k=1, M=mass, which="SM")
evecs = -evecs

print("Condition c^T M c = ", evecs.T @ mass @ evecs, ".")
print(x)

ax = plt.axes(projection='3d')
ax.scatter3D(x, y, evecs)
plt.show()