import numpy as np

def kron_fast(A, B, type):
    # Matrix dimensions
    p, q = A.shape
    qn, m = B.shape
    n = qn // q
    
    # Compute
    if type == 0:
        C = np.reshape(np.dot(np.reshape(B.T, (n * m, q)), A.T), (m, p * n)).T
    elif type == 1:
        C = np.reshape(np.dot(A, np.reshape(B, (q, n * m))), (p * n, m))
    
    return C