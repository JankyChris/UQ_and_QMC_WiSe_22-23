# Solution to Exercise 5.3 of the course
# Uncertainty Quantification and Quasi-Monte Carlo
# by Christoph Jankowsky

import numpy as np
import matplotlib.pyplot as plt

def triangulate(level):
    h = 1/2**level
    N = int(2 / h)

    x = np.linspace(0,2,N+1)
    y = np.linspace(0,2,N+1)
    nodes = []
    for i in x:
        for j in y:
            if (i > 1.0) and (j > 1.0):
                pass
            else:
                nodes.append((i,j))

    element = []
    for (x, y) in nodes:
        if (x == 2.0) or (y == 2.0):
            pass
        elif (x >= 1.0 and y == 1.0) or (x == 1.0 and y >= 1.0):
            pass
        else:
            element.append([(x, y), (x+h, y), (x, y+h)])

        if (x == 0.0) or (y == 0.0):
            pass
        else:
            element.append([(x, y), (x-h, y), (x, y-h)])
    
    interior = []
    for (x, y) in nodes:
        if x == 0.0 or x == 2.0 or y == 0.0 or y == 2.0:
            pass
        elif (x >= 1.0 and y == 1.0) or (x == 1.0 and y >= 1.0):
            pass
        else:
            interior.append((x, y))
        
    

    return nodes, element, interior

def plot(nodes, element, interior):
    Purple = '#452399'
    Red = '#e33667'

    ax = plt.subplot()

    for triangle in element:
        tri = plt.Polygon(triangle, fill=None, alpha=1)
        ax.add_patch(tri)

    x_nodes = [nodes[i][0] for i in range(len(nodes))]
    y_nodes = [nodes[i][1] for i in range(len(nodes))]
    ax.scatter(x_nodes, y_nodes, c=Purple, s=15)

    x_int = [interior[i][0] for i in range(len(interior))]
    y_int = [interior[i][1] for i in range(len(interior))]
    ax.scatter(x_int, y_int, c=Red, s=15)

    plt.show()

if __name__ == "__main__":
    levels = [i for i in range(1, 4)]

    for level in levels:
        nodes, element, interior = triangulate(level)
        plot(nodes, element, interior)
