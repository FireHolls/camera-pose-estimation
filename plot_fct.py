import matplotlib.pyplot as plt
import numpy as np

def plot_points_only(px_true, pts3d, P_est, title="True vs. Reprojected Points"):
    """
    px_true: (2, N) True pixel coordinates (Ground Truth)
    pts3d:   (3, N) True 3D world points
    P_est:   (3, 4) The estimated projection matrix K[R|t]
    """
    
    # 1. Reproject 3D points using the estimated P matrix
    # Convert pts3d to homogeneous (4, N)
    pts3d_h = np.vstack((pts3d, np.ones((1, pts3d.shape[1]))))
    px_reproj_h = P_est @ pts3d_h
    
    # Normalize by the last row (Z) to get final 2D pixel coordinates (2, N)
    px_reproj = px_reproj_h[:2, :] / px_reproj_h[2, :]

    # 2. Plotting
    plt.figure(figsize=(5, 4))
    
    # Plot Ground Truth points (Blue Circles)
    plt.scatter(px_true[0, :], px_true[1, :], c='blue', marker='o', s=40,
                label='Ground Truth (True P)', alpha=0.6)
    
    # Plot Reprojected points (Red Crosses)
    plt.scatter(px_reproj[0, :], px_reproj[1, :], c='red', marker='x', s=60,
                label='Reprojected (Estimated P)', alpha=0.9)

    plt.title(title)
    plt.xlabel("Pixel Coordinate X")
    plt.ylabel("Pixel Coordinate Y")
    plt.legend()
    
    # Invert Y axis to match image coordinate systems (0,0 at top-left)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()