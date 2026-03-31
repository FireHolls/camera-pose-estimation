import numpy as np
from SVD import SVD

def get_R_t_from_F(F, K):
    """
    Function to retrieve the rotational matrix and translational matrix 
    given the fundamental matrix, assuming P = [I|0] and P' = [R|t]
    Input:
        - F: Fundamental matrix (3x3, np.array)
        - K: Intrinsic matrix (3x3, np.array)
    Ouput:
        - t:  Translational matrix
        - R1: Rotational matrix (option 1)
        - R2: Rotational matrix (option 2)
    """

    E = K.T@F@K # Essential matrix
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    Z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]
    ])
    U, Sigma, V = SVD(E)
    t = U[:, 2]
    R1 = U@W@V.T
    R2 = U@W.T@V.T
    return t, R1, R2