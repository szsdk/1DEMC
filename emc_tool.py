import numpy as np

def get_thetaAB(a, b, n):
    d = np.abs(a-b)
    idx = d>(n/2.0)
    d[idx] = n-d[idx]
    return d

