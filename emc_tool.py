import numpy as np

def get_thetaAB(a, b, n):
    d = np.abs(a-b)
    idx = d>(n/2.0)
    d[idx] = n-d[idx]
    return d

def match(A,Bt):
    B = Bt * np.sum(A)/ np.sum(Bt)
    n_min = 0
    val = 1e100
    for i in range(len(A)):
        val_tmp = np.sum((np.roll(B, i)-A)**2)
        if val_tmp < val:
            n_min = i
            val = val_tmp
    return np.roll(B,n_min), n_min
