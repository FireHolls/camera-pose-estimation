import numpy as np
from eight_points.triangulation import triangulate
from simulation.projection import project_points

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

def P_estimation(t, R1, R2, K, s = None):
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
    if s is None:
        s = 1
        
    P_est1 = K@np.hstack((R1, s*t)) # Rotation 1, Positive translation
    P_est2 = K@np.hstack((R1, -s*t)) # Rotation 1, Negative translation
    P_est3 = K@np.hstack((R2, s*t)) # Rotation 2, Positive translation
    P_est4 = K@np.hstack((R2, -s*t)) # Rotation 2, Negative translation
    # Stacking all the possible projection matrices
    P_est = np.stack([P_est1, P_est2, P_est3, P_est4])
    return P_est

def parallax(P2_all, K, pt1, pt2):
    R1 = np.eye(3)
    t1 = np.zeros((3,1))
    P1 = K@np.hstack((R1, t1))
    K_inv = np.linalg.inv(K)
    best_R2 = None
    best_t2 = None
    best_P2 = None
    max_positive_depths = -1
    for i in range(4):
        P2 = P2_all[i,:,:]
        pt3D = triangulate(P1, P2, pt1, pt2)
        P2_noK = K_inv@P2
        R2 = P2_noK[:, :3]
        t2 = P2_noK[:, 3].reshape(3, 1)
        _, d1 = project_points(pt3D, K, R1, t1)
        _, d2 = project_points(pt3D, K, R2, t2)
        valid_mask = (d1 > 0) & (d2 > 0)
        valid_count = np.sum(valid_mask)
        if valid_count > max_positive_depths:
            max_positive_depths = valid_count
            best_R2 = R2
            best_t2 = t2
            best_P2 = P2
    print(max_positive_depths)
    return best_R2, best_t2, best_P2

def find_scaling_factor(P2, K, pts1, pts2, pts3D):
    P1 = K@np.hstack((np.eye(3), np.zeros((3, 1))))
    pts3D_triag = triangulate(P1, P2, pts1, pts2)
    scale = []
    for i in range(4):
        for j in range(i):
            dist_truth = np.linalg.norm(pts3D[:,i] - pts3D[:,j])
            dist_triag = np.linalg.norm(pts3D_triag[:, i] - pts3D_triag[:, j])
            s_f = dist_truth / dist_triag
            scale.append(s_f)
    s = np.median(scale)
    return s



