import numpy as np
import math
from scipy.linalg import block_diag

"""
These algorithms are based on the algorithms explained in [Golub, Van Loan, "Matrix Computations,
third edition", 1983]
"""

def Householder_vector(x):
    """
    This function calculates the Householder vector x. Given a vector x of size n, it computes 
    a vector v of size n with v(1) = 1 and a scalar beta such that P = I - beta*v*v' is orthogonal
    and Px = ||x||e1 (Algo 5.1.1)
    Input:
        - x: A vector of size n
    Output:
        - v: A vector of size n
        - beta: A scalar
    """
    n = len(x)
    sigma = x[1:n].T@x[1:n]
    v = np.concatenate(([1], x[1:n]))
    if sigma == 0:
        beta = 0
    else:
        miu = math.sqrt(x[0]**2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - miu
        else:
            v[0] = -sigma/(x[0] + miu)
        beta = 2*v[0]**2/(sigma + v[0]**2)
        v = v/v[0]
    return v, beta


def Householder_Bidiagonalization(A):
    """
    This function overwrites a matrix A (m x n) where m >=n with the matrix a matrix B which 
    is upper diagonal where B = U'*A*V, where U and V are orthogonal (Algo 5.4.2)
    Input:
        - A: A m x n matrix (np.array) with m >= n
    Ouput:
        - A: Upper diagonal matrix of size m x n (np.array)
        - U: Orthogonal matrix of size m x m (np.array)
        - V: Orthogonal matrix of size n x n (np.array)
    """
    A = np.array(A, dtype=float, copy=True)
    rows, columns = A.shape

    U = np.eye(rows)
    V = np.eye(columns)

    for j in range(columns):
        # Left Householder
        v, beta = Householder_vector(A[j:rows, j])
        H = np.eye(rows - j) - beta * np.outer(v, v)

        A[j:rows, j:columns] = H @ A[j:rows, j:columns]
        U[:, j:] = U[:, j:] @ H

        # Right Householder
        if j < columns - 2:
            v, beta = Householder_vector(A[j, j+1:columns])
            G = np.eye(columns - j - 1) - beta * np.outer(v, v)

            A[j:rows, j+1:columns] = A[j:rows, j+1:columns] @ G
            V[:, j+1:] = V[:, j+1:] @ G

    return A, U, V

def givens(a, b):
    """
    The function computes c = cos(theta) and s = sin(theta) such that 
    [ c  s      [a    [r  
     -s  c]^T *  b] =  0] (Algo 5.1.3)
     Input:
        - a: Scalar
        - b: Scalar
    Output:
        - c: Scalar
        - s: Scalar
    """
    if b == 0:
        c, s = 1.0, 0.0
        return c, s
    else:
        if abs(b) > abs(a):
            tau = -a/b
            s = 1/math.sqrt(1 + tau**2)
            c = s*tau
        else:
            tau = -b/a
            c = 1/math.sqrt(1 + tau**2)
            s = c*tau  
        return c, s

def givens_matrix(c, s, i, k, size):
    """
    This function computes the Givens rotational matrix (formula 5.1.7)
    Input:
        - c: Scalar determined from function "givens"
        - s: Scalar determined from function "givens"
        - i, k: Intergers - positions of c and s in the Givens matrix
        - size: Interger - size of the Givens rotational square matrix
    Output:
        - G: Givens matrix of size: size x size with c and s implemented
    """
    G = np.eye(size)
    G[i, i], G[i, k], G[k, i], G[k, k] = c, s, -s, c
    return G

def QR_decomposition(B):
    """
    This function overwrites the bidiagonal matrix B with a new matrix U'BV which is nearly 
    diagonal. (Algo 8.3.2)
    Input:
        - B: Bidiagonal matrix (np.array)
    Ouput:
        - B: The new "diagonal matrix"
        - U, V: The orthogonal matrices which turn B into a diagonal matrix
    """
    T = B.T@B
    n = T.shape[0]
    d = (T[n-2, n-2] - T[n-1, n-1])/2
    miu = T[n-1, n-1] - T[n-1, n-2]**2/(d + (d*d/d)*math.sqrt(d**2*T[n-1, n-2]**2))
    y = T[0, 0] - miu
    z = T[1, 0]
    U = np.eye(n)
    V = np.eye(n)
    for k in range(n-1):
        c, s = givens(y, z)
        G = givens_matrix(c,s, k, k+1, B.shape[1])
        B = B@G
        V = V@G
        y, z = B[k, k], B[k+1, k]
        c, s = givens(y, z)
        G = givens_matrix(c,s, k, k+1, B.shape[1])
        B = G.T@B
        U = U@G
        if k < n - 2:
            y, z = B[k, k+1], B[k, k+2]
    return B, U, V

def check_condition_diagonal(B):
    """
    This function decomposes a bidiagonal B of size n x n into three sub-matrices, B11, B22, B33 
    where B11 is of size p x p, B22, of size n - p - q x n - p - q and B33, of size q x q. And, 
    find the largest q and the smallest p such that B33 is diagonal and B22 has nonzero 
    superdiagonal. (Condition in algo 8.6.2) 
    Input:
        - B: A bidiagonal matrix (np.array)
    Output:
        - q, p: Largest and smallest value of q and p respectively such that the condition above is achieved
        - B22: A bidiagonal or diagonal matrix (np.array) or None, if the conditions are not achieved
    """
    n = B.shape[1]
    for q in range(n-1, -1, -1):
        for p in range(0, n-q-1):
            if p != 0:
                B11 = B[0:p, 0:p]
                B22 = B[p:n-q, p:n-q]
                B33 = B[n-q:n, n-q:n]
            else:
                B22 = B[0:n-q, 0:n-q]
                B33 = B[n-q:n, n-q:n]
            B33_superdiag = np.diag(B33, k=1)
            B22_superdiag = np.diag(B22, k=1)
            if np.all(B33_superdiag == 0):
                if (len(B22_superdiag) > 0 and np.all(B22_superdiag != 0)):
                    return q, p, B22
    return None, None, None

def SVD(A, epsilon=1e-12, max_iter=1000):
    """
    This function executes the Singular Value Decomposition (SVD) of a matrix A of size m x n,
    where m >= n. It computes U, an orthogonal matrix, Sigma a "diagonal" matrix and V, an 
    orthogonal matrix (Algo 8.6.2).
    Input:
        - A: Matrix of size m x n (np.array)
    Output:
        - U: Orthogonal matrix of size m x m (np.array)
        - Sigma: Diagonal matrix of size m x n (np.array)
        - V: Orthogonal matrix of size n x n (np.array)
    """
    B, U0, V0 = Householder_Bidiagonalization(A)
    m, n = B.shape
    U_iter = np.eye(m)
    V_iter = np.eye(n)
    for k in range(max_iter):
        for i in range(n-1):
            if abs(B[i, i+1]) <= epsilon*(abs(B[i, i]) + abs(B[i+1, i+1])):
                B[i, i+1] = 0
        q, p, B22 = check_condition_diagonal(B)
        if B22 is None:
            break
        if q < n-1:
            if np.any(np.diag(B22) == 0):
                for i in range(B22.shape[0]):
                    if B22[i, i] == 0 and i != B22.shape[0]-1:
                        B22[i, i+1] = 0
                        B[p + i, p + i + 1] = 0
            else:
                B22, U_loc, V_loc = QR_decomposition(B22)
                Ustep = block_diag(np.eye(p), U_loc, np.eye(q + m - n))
                Vstep = block_diag(np.eye(p), V_loc, np.eye(q))
                B = Ustep.T @ B @ Vstep
                U_iter = U_iter @ Ustep
                V_iter = V_iter @ Vstep
    # Final orthogonal factors
    U = U0 @ U_iter
    V = V0 @ V_iter

    # Make singular values nonnegative
    s = np.diag(B[:, :]).copy()
    signs = np.where(s < 0, -1.0, 1.0)

    U[:, :n] = U[:, :n] * signs
    s = np.abs(s)

    # Sort singular values in descending order
    order = np.argsort(-s)
    s = s[order]

    U_sorted = U.copy()
    U_sorted[:, :n] = U[:, :n][:, order]
    V_sorted = V[:, order]

    Sigma = np.zeros((m, n))
    Sigma[:n, :n] = np.diag(s)

    return U_sorted, Sigma, V_sorted