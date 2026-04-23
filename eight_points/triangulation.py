import numpy as np
import math

def normalize(points):
    """
    This function normalizes the set of points before applying DLT and 8-point algorithm
    to improve the accuracy.
    Input:
        - points: The set of 2D points (N, 3) - np.array
    Output:
        - T: Normalization matrix of size 3x3 (np.array)
    """
    N = points.shape[0]
    centroid = 1/N * np.sum(points[:, :2], axis=0) #Calculate centroid
    # Calculate shifted points 
    shifted_points = points[:, :2] - centroid
    d_avg = np.mean(np.sqrt(shifted_points[:, 0]**2 + shifted_points[:, 1]**2))
    if d_avg == 0:
        raise ValueError("All points are identical; normalization is undefined.")
    s = math.sqrt(2)/d_avg #Scaling factor
    T = np.array([
        [s, 0, -s * centroid[0]],
        [0, s, -s * centroid[1]],
        [0, 0, 1]
    ])
    return T

def triag_system(P1, P2, pts1, pts2):
    """
    This function build the system equation A in which AX = 0 and x = PX and x' = P1X
    Input: 
        - P: Projection matrix of the first camera (np.array of siez 3x4)
        - P1: Projeciton matrix of the second camera (np.array of size 3x4)
        - pts1: The coordinate points from the first image (np.array of size 1x2)
        - pts2: The coordinate points from the second image (np.array of size 1x2)
    Output:
        - A: The linear system of equation (np.array of size 4x4)
    """
    A = np.zeros((4,4))
    A[0,:] = pts1[0] * P1[2,:] - P1[0,:]
    A[1,:] = pts1[1] * P1[2,:] - P1[1,:]
    A[2,:] = pts2[0] * P2[2,:] - P2[0,:]
    A[3,:] = pts2[1] * P2[2,:] - P2[1,:]

    return A

def triangulate(P1, P2, pts1, pts2): 
    """
    This function find the 3D points such that x = PX and x' = P1X
    Input: 
        - P: Projection matrix of the first camera (np.array of siez 3x4)
        - P1: Projeciton matrix of the second camera (np.array of size 3x4)
        - pts1: The coordinate points from the first image (np.array of size Nx3)
        - pts2: The coordinate points from the second image (np.array of size Nx3)
    Output:
        - X: 3D point (np.array of size Nx3)
    """

    #1 : Normalization
    T1 = normalize(pts1)
    T2 = normalize(pts2)
    points1 = pts1.copy()
    points2 = pts2.copy()
    norm1 = (T1 @ points1.T).T
    norm2 = (T2 @ points2.T).T
    P1_norm = T1 @ P1
    P2_norm = T2 @ P2

    #2: Calculate 3D points
    N = points1.shape[0]
    X = np.zeros((N, 4))
    for i in range(N):
        A = triag_system(P1_norm, P2_norm, norm1[i,:2], norm2[i,:2])
        U, S, Vh = np.linalg.svd(A)
        X[i,:] = Vh[-1, :]
        X[i, :] = X[i, :] / X[i, 3]
    
    return X[:, :3]

