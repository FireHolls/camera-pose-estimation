import numpy as np
from SVD import SVD
from normalization import normalize

def linear_eq(points1, points2):
    length = min(points1.shape[0], points2.shape[0])
    A = np.zeros((length, 9))
    for i in range(length):
        A[i, 0] = points2[i, 0]*points1[i, 0]
        A[i, 1] = points2[i, 0]*points1[i, 1]
        A[i, 2] = points2[i, 0]
        A[i, 3] = points2[i, 1]*points1[i, 0]
        A[i, 4] = points2[i, 1]*points1[i, 1]
        A[i, 5] = points2[i, 1]
        A[i, 6] = points1[i, 0]
        A[i, 7] = points1[i, 1]
        A[i, 8] = 1
    return A

def eight_point(points1, points2):

    length = min(points1.shape[0], points2.shape[0])
    #1: Normalization
    T1 = normalize(points1)
    T2 = normalize(points2)
    norm1 = (T1 @ points1.T).T
    norm2 = (T2 @ points2.T).T

    #2: Find the fundamental matrix of the normalized points
    A_hat = linear_eq(norm1, norm2)
    U, Sigma, V = SVD(A_hat)
    F_hat = np.array([
        [V[0,8], V[1,8], V[2,8]],
        [V[3,8], V[4,8], V[5,8]],
        [V[6,8], V[7,8], V[8,8]]
    ])

    #3: Replace F_hat but Fprime_hat such that det Fprime_hat = 0
    U, Sigma, V = SVD(F_hat)
    Sigma[2,2] = 0
    Fprime_hat = U@Sigma@V.T
    
    #4: Denormalize
    F = T2.T@Fprime_hat@T1

    return F





