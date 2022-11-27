import numpy as np
import time
from joblib import Parallel, delayed

def foo(i):
    A = np.random.uniform(low=0.0, high=1.0, size=(150,150))
    y = np.random.uniform(low=0.0, high=1.0, size=(150,1))

    sol = np.linalg.solve(A, y)

    return np.sum(sol)

if __name__ == "__main__":

    # sequential:
    t0 = time.time()
    for i in range(10000):
        foo(i)
    t1 = time.time()
    time_sequential = t1-t0
    print("Time for sequential loop:", "{0:.2f}".format(time_sequential), "s.")

    # parallel:
    t0 = time.time()
    results = Parallel(n_jobs=-1)(delayed(foo)(i) for i in range(10000))
    t1 = time.time()
    time_parallel = t1-t0
    print("Time for parallel loop:  ", "{0:.2f}".format(time_parallel), "s.")