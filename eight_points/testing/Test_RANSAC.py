import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from simulation.camera_model import get_K, get_camera_pose
from simulation.scene_generator import PointsGenerator
from simulation.projection import  project_points,filter_visible
from simulation.dlt_verification import reprojection_error
from eight_points.eight_point_agl import eight_point
from eight_points.Retrieve_P import get_R_t_from_epipolar, P_estimation, parallax, find_scaling_factor
from eight_points.triangulation import triangulate
from plot_fct import plot_points
import matplotlib.pyplot as plt
from eight_points.RANSAC import RANSAC

def score_F(F, px1, px2, threshold=3.84):
    """
    Symmetric Sampson distance score for fundamental matrix F (ORB-SLAM style).

    Sampson distance is the first-order approximation of the reprojection error
    under the epipolar constraint x2ᵀFx1 = 0 :

      d²_sampson = (x2ᵀFx1)² / (‖Fx1‖²_top2 + ‖Fᵀx2‖²_top2)

      S_F = Σ max(0, T - d²_sampson)

    threshold : chi² at 95% for 1 DOF = 3.84  (epipolar constraint is 1D)

    Returns: float  (higher = better fit)
    """
    N = px1.shape[1]
    h1 = np.vstack([px1, np.ones((1, N))])
    h2 = np.vstack([px2, np.ones((1, N))])

    Fx1  = F @ h1                                              # (3, N)
    Ftx2 = F.T @ h2                                           # (3, N)

    num      = np.sum(h2 * Fx1, axis=0) ** 2                  # (x2ᵀFx1)²
    denom    = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
    d_samp   = num / (denom + 1e-12)
    score = float(np.sum(np.maximum(0, threshold - d_samp)))
    inlier_mask = d_samp < threshold
    return score, inlier_mask


# 1 Génération de la scène 
bounds = np.array([-2.0, 2.0, -1.5, 1.5, 10, 20])
pts3d  = PointsGenerator(nbPoints=50, seed=43, bounds=bounds)

# 2 Définition des caméras 
K = get_K()
R1, t1 = np.eye(3), np.zeros(3)
R2, t2 = get_camera_pose(rz=10, tx=4, ty = 2)

# 3 Projection 
px1, d1 = project_points(pts3d, K, R1, t1)
px2, d2 = project_points(pts3d, K, R2, t2)
print(px2.shape)
W = 1920 # Image width
H = 1080 # Image height
missmatch = 10
for i in range(missmatch):
    px2[0, i] = np.random.uniform(0, W)  
    px2[1, i] = np.random.uniform(0, H)

# 4 Filtrage 
vis = filter_visible(px1, d1) & filter_visible(px2, d2)
pts3d_vis = pts3d[:, vis]
print(f"Points visibles dans les deux caméras : {vis.sum()} / {pts3d.shape[1]}")
px1_vis = px1[:, vis]
px2_vis = px2[:, vis]

#5 Turn the points to homogenous 2D points

ransac_solver = RANSAC(
    s=8, 
    epsilon=0.2, 
    score_fct=score_F,
    model_fct=eight_point, 
    px1=px1_vis, 
    px2=px2_vis
)
F, mask = ransac_solver.execute_RANSAC()

clean_px1 = px1_vis[:, mask]
clean_px2 = px2_vis[:, mask]
print(mask)
print(f"RANSAC kept {clean_px1.shape[1]} points out of {px1_vis.shape[1]}.")