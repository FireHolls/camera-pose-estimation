import numpy as np

def get_R_t_from_epipolar(F, K = None):
    """
    Function to retrieve the rotational matrix and translational matrix 
    given the fundamental matrix (if K is not None) or given the essential matrix (if K is None), 
    assuming P = [I|0] and P' = [R|t]
    Input:
        - F: Fundamental matrix (3x3, np.array) - K is not None OR the essential matrix - if K is None
        - K: Intrinsic matrix (3x3, np.array) - Optional
    Ouput:
        - t:  Translational matrix
        - R1: Rotational matrix (option 1)
        - R2: Rotational matrix (option 2)
    """

    if K is not None:
        # Calculate Essential matrix from Fundamental matrix and Intrinsics
        E = K.T@F@K 
    else:
        E = F.copy()
    
    # Standard orthogonal matrices used for decomposing E
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    # Decompose the Essential matrix
    U, S, Vh = np.linalg.svd(E)

    # The translation is the last column of U (left null space)
    t = U[:, 2].reshape(-1, 1) 
    
    # Two possible rotation matrices
    R1 = U@W@Vh
    R2 = U@W.T@Vh

    # Ensure they are valid rotation matrices (det = +1, not -1, to prevent reflections)
    if np.linalg.det(R1) < 0: R1 = -R1
    if np.linalg.det(R2) < 0: R2 = -R2

    return t, R1, R2

def P_estimation(t, R1, R2, K, s):
    """
    Function which assembles the four possible solution of the projection matrices (P' = K[R | t]).
    Input:
        - t: Estimate of the translation unit vector (np.array of size 3x1)
        - R1: Estimate of the rotation matrix - option n°1 (np.array of size 3x3)
        - R2: Estimate of the rotation matrix - option n°2 (np.array of size 3x3)
        - K: Intrinsic parameter matrix (np.array of size 3x3)
        - s: Scaling factor to convert the unit vector 't' to actual distance
    Output:
        - P_est: Stack containing all the possible projection matrices (np.array of size 4x3x4)
    """

    P_est1 = K@np.hstack((R1, s*t)) # Rotation 1, Positive translation
    P_est2 = K@np.hstack((R1, -s*t)) # Rotation 1, Negative translation
    P_est3 = K@np.hstack((R2, s*t)) # Rotation 2, Positive translation
    P_est4 = K@np.hstack((R2, -s*t)) # Rotation 2, Negative translation

    # Stacking all the possible projection matrices
    P_est = np.stack([P_est1, P_est2, P_est3, P_est4])
    return P_est