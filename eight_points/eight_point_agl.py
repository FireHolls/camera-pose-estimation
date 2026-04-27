import numpy as np
import math
from scipy.linalg import block_diag
from scipy.optimize import least_squares

def normalize(points):
    """
    This function normalizes the set of points before applying DLT and 8-point algorithm
    to improve the accuracy.
    Input:
        - points: The set of 2D points (3, N) - np.array
    Output:
        - T: Normalization matrix of size 3x3 (np.array)
    """
    N = points.shape[1]
    centroid = 1/N * np.sum(points[:2, :], axis=1) #Calculate centroid
    # Calculate shifted points 
    shifted_points = points[:2, :] - centroid.reshape(2, 1)
    d_avg = np.mean(np.sqrt(shifted_points[0, :]**2 + shifted_points[1, :]**2))
    if d_avg == 0:
        raise ValueError("All points are identical; normalization is undefined.")
    s = math.sqrt(2)/d_avg #Scaling factor
    T = np.array([
        [s, 0, -s * centroid[0]],
        [0, s, -s * centroid[1]],
        [0, 0, 1]
    ])
    return T

def linear_eq(points1, points2):
    """
    Function to construct the matrix A used to solve the epipolar constraint equation x'TFx = 0, 
    where Af = 0
    Input:
        - points1: 2Dpoints of the first image (np.array of size 2xN or 3xN)
        - points2: 2Dpoints of the second image (np.array of size 2xN or 3xN)
    Output:
        - A: The linear system equation (np.array of size Nx9)
    """
    # Determine the number of point correspondences
    length = min(points1.shape[1], points2.shape[1])
    
    # Initialize the design matrix A (N x 9) for the equation Af = 0
    A = np.zeros((length, 9))
    for i in range(length):
        A[i, 0] = points2[0, i]*points1[0, i]
        A[i, 1] = points2[0, i]*points1[1, i]
        A[i, 2] = points2[0, i]
        A[i, 3] = points2[1, i]*points1[0, i]
        A[i, 4] = points2[1, i]*points1[1, i]
        A[i, 5] = points2[1, i]
        A[i, 6] = points1[0, i]
        A[i, 7] = points1[1, i]
        A[i, 8] = 1
    return A

def eight_point(pts1, pts2, K1 = None, K2 = None):
    """
    Function to execute the normalized 8-points algorithm to determine the fundamental matrix F or 
    the essential matrix if K1 and K2 are not None. 
    Input:
        - pts1: 2D homogenous points of the first image - pixel coordinates (np.array of size Nx3)
        - pts2: 2D homogenous points of the second image - pixel coordinates (np.array of size Nx3)
        - K1: Intrinsic parameter of the first camera (np.array of size 3x3) - optional
        - K2: Intrinsic parameter of the second camera (np.array of size 3x3) - optional
    Output:
        - F: Fundamental matrix (np.array 3x3) OR if K1 and K2 are not None: Essential matrix
    """
    points1 = pts1.copy()
    points2 = pts2.copy()
    # Function computes the essential matrix instead of the fundamental matrix
    if K1 is not None and K2 is not None: 
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        points1 = K1_inv @ points1 
        points2 = K2_inv @ points2

    #1: Normalization: increase stability
    T1 = normalize(points1) # Normalization matrix
    T2 = normalize(points2)
    if K1 is not None and K2 is not None: 
        norm1 = T1 @ points1
        norm2 = T2 @ points2
    else:
        norm1 = T1 @ points1
        norm2 = T2 @ points2

    #2: Find the fundamental matrix of the normalized points
    A_hat = linear_eq(norm1, norm2) # Normalized linear system where A_hat*f_hat = 0
    U, S, Vh = np.linalg.svd(A_hat)
    F_hat = Vh[8, :].reshape(3, 3)

    #3: Replace F_hat but Fprime_hat such that det Fprime_hat = 0
    U1, S1, Vh1 = np.linalg.svd(F_hat)
    S1[2] = 0
    Fprime_hat = U1@np.diag(S1)@Vh1
    
    #4: Mminimize algebriac error and denormalize
    F_hat_opt = min_alg_error(Fprime_hat, A_hat)
    F = T2.T@F_hat_opt@T1

    return F

def min_alg_error(F0, A):
    """
    Function which calculate the fundamental matrix by minimizing the algebraic error.
    Input:
        - F0: Inital guess of the fundamentral matrix (np.array of size 3x3)
        - A: Linear system equation (np.array of size Nx9)
    Ouput:
        - F_opt: Optimal fundamental matrix (np.array of size 3x3)
    """
    
    # Find the right null space e0 of F0 (the initial epipole guess)
    U, S, Vh = np.linalg.svd(F0)
    e0 = Vh[-1, :] #Last row of Vh
    tracker = {'prev_epsilon': None} # Tracker to prevent the SVD sign-flipping issue between LM iterations
    
    # Run Levenberg-Marquardt to optimize the epipole coordinates
    result = least_squares(
        fun=residual_fun, # Cost function
        x0=e0, # Initial guess
        args=(A, tracker),  
        method='lm',
        xtol=1e-15,  
        ftol=1e-15   
    )

    # Reconstruct the optimized Fundamental matrix using the optimal epipole
    e_opt = result.x
    ex = np.array([
        [0,        -e_opt[2],  e_opt[1]],
        [e_opt[2],  0,        -e_opt[0]],
        [-e_opt[1], e_opt[0],  0       ]
    ])
    E_opt = block_diag(ex, ex, ex)
    
    # Get final 9-element vector and reshape to 3x3
    f_opt = constrained_min(A, E_opt)
    F_opt = f_opt.reshape(3, 3)
    
    return F_opt

def constrained_min(A, G):
    """
    Function which solves x that minimizes ||Ax|| subject to x = G*x_hat and ||x|| = 1
    Input: 
        - A: The linear system equation (np.array of size Nx9)
        - G: Block-diagonal constraint matrix built from the skew-symmetric matrix of the epipole
             (np.array of size 9x9)
    Ouput:
        - x: Best-fit Fundamental Matrix for that specific epipole (vector of size 9)
    """
    # Find the valid parameter subspace defined by G
    U, S, Vh = np.linalg.svd(G)
    r = np.linalg.matrix_rank(G) # Determine true degrees of freedom
    U_prime = U[:,:r] # Extract basis for the subspace
    
    # Solve the reduced system
    _, _, Vh2 = np.linalg.svd(A @ U_prime)
    x_prime = Vh2[-1, :] # The minimizer in the reduced space
    # Project back to the original 9D space
    x = U_prime@x_prime
    return x

def residual_fun(e, A, state):
    """
    Cost function to be implemented in the LM algorithm which calculates how much 
    error is produced by a specific guess for the epipole.
    Input:
        - e: Current guess of the epipole coordinates (vector of size 3)
        - A: Linear system equation (np.array Nx9)
        - state: Tracker for the previous error (dictionnary)
    Output:
        - epsilon: Algebraic error residuals (np.array of size 1x1)
    """
    # Objective function evaluated at every LM iteration

    # Build skew-symmetric matrix of current epipole guess  
    ex = np.array([
        [0,    -e[2],  e[1]],
        [e[2],  0,    -e[0]],
        [-e[1], e[0],  0   ]
    ])
    
    # Constraint matrix E ensuring F*e = 0
    E = block_diag(ex, ex, ex)
    f = constrained_min(A, E)

    # Compute current algebraic error vector
    epsilon = A@f

    # Enforce smooth sign variation to prevent SVD from flipping directions randomly
    if state['prev_epsilon'] is not None:
        dot_product = np.dot(epsilon, state['prev_epsilon'])
        if dot_product < 0:
            epsilon = -epsilon # Flip sign if pointing the opposite way from last step
    state['prev_epsilon'] = epsilon.copy() # Save current error for the next iteration

    return epsilon





