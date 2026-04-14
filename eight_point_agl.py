import numpy as np
from normalization import normalize
from scipy.linalg import block_diag
from scipy.optimize import least_squares

def linear_eq(points1, points2):
    """
    Function to construct the matrix A used to solve the epipolar constraint equation x'TFx = 0, 
    where Af = 0
    Input:
        - points1: 2Dpoints of the first image (np.array of size Nx2 or Nx3)
        - points2: 2Dpoints of the second image (np.array of size Nx2 or Nx3)
    Output:
        - A: The linear system equation (np.array of size Nx9)
    """
    # Determine the number of point correspondences
    length = min(points1.shape[0], points2.shape[0])
    
    # Initialize the design matrix A (N x 9) for the equation Af = 0
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

def eight_point(points1, points2, K1 = None, K2 = None):
    """
    Function to execute the normalized 8-points algorithm to determine the fundamental matrix F or 
    the essential matrix if K1 and K2 are not None. 
    Input:
        - points1: 2D homogenous points of the first image - pixel coordinates (np.array of size Nx3)
        - points2: 2D homogenous points of the second image - pixel coordinates (np.array of size Nx3)
        - K1: Intrinsic parameter of the first camera (np.array of size 3x3) - optional
        - K2: Intrinsic parameter of the second camera (np.array of size 3x3) - optional
    Output:
        - F: Fundamental matrix (np.array 3x3) OR if K1 and K2 are not None: Essential matrix
    """
    # Function computes the essential matrix instead of the fundamental matrix
    if K1 is not None and K2 is not None: 
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        points1 = K1_inv@points1.T # Convert the points from the image frame to the camera frame
        points2 = K2_inv@points2.T

    #1: Normalization: increase stability
    T1 = normalize(points1) # Normalization matrix
    T2 = normalize(points2)
    norm1 = (T1 @ points1.T).T
    norm2 = (T2 @ points2.T).T

    #2: Find the fundamental matrix of the normalized points
    A_hat = linear_eq(norm1, norm2) # Normalized linear system where A_hat*f_hat = 0
    U, S, Vh = np.linalg.svd(A_hat)
    F_hat = Vh[-1, :].reshape(3, 3)

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





