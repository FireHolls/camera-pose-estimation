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
    for i in range(N): #Translation
        points[i, 0] = points[i, 0] - centroid[0]
        points[i, 1] = points[i, 1] - centroid[1]
    d_avg = np.mean(np.sqrt(points[:, 0]**2 + points[:, 1]**2))
    if d_avg == 0:
        raise ValueError("All points are identical; normalization is undefined.")
    s = math.sqrt(2)/d_avg #Scaling factor
    T = np.array([
        [s, 0, -s * centroid[0]],
        [0, s, -s * centroid[1]],
        [0, 0, 1]
    ])
    return T

