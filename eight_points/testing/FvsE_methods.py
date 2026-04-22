import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from simulation.camera_model import get_K, get_camera_pose
from simulation.scene_generator import PointsGenerator
from simulation.projection import  project_points,filter_visible
from simulation.dlt_verification import reprojection_error
from eight_points.eight_point_agl import eight_point
from eight_points.Retrieve_P import get_R_t_from_epipolar, P_estimation
from plot_fct import plot_points
import matplotlib.pyplot as plt

# 1 Génération de la scène 
bounds = np.array([-2.0, 2.0, -1.5, 1.5, 4.0, 10.0])
pts3d  = PointsGenerator(nbPoints=200, seed=42, bounds=bounds)

# 2 Définition des caméras 
K = get_K()
R1, t1 = np.eye(3), np.zeros(3)
R2, t2 = get_camera_pose(ry=8, tx=0.4, ty=0.0, tz=0.0)
norm_t2 = np.linalg.norm(t2)

# 3 Projection 
px1, d1 = project_points(pts3d, K, R1, t1)
px2, d2 = project_points(pts3d, K, R2, t2)

# 4 Filtrage 
vis = filter_visible(px1, d1) & filter_visible(px2, d2)
pts3d_vis = pts3d[:, vis]
print(f"Points visibles dans les deux caméras : {vis.sum()} / {pts3d.shape[1]}")
px1_vis = px1[:, vis]
px2_vis = px2[:, vis]

#5 Turn the points to homogenous 2D points
points1 = np.vstack((px1_vis, np.ones((1, px1_vis.shape[1]))))
points2 = np.vstack((px2_vis, np.ones((1, px2_vis.shape[1]))))
print(points1.shape)

#6 Retrieve the translation vector and rotation matrices
F = eight_point(points1.T, points2.T) # Calculate F first then deduce E
tf, R_1f, R_2f = get_R_t_from_epipolar(F, K = K) 
E = eight_point(points1.T, points2.T, K1 = K, K2 = K) # Calculate E directly
te, R_1e, R_2e = get_R_t_from_epipolar(E, K = None)

#7 Estrimate the projection matrix
P_estf = P_estimation(tf, R_1f, R_2f, K, norm_t2) # First method
P_este = P_estimation(te, R_1e, R_2e, K, norm_t2) # Second method

#8 Determine reprojection error

rmsef = np.zeros((4, 1))
rmsee = np.zeros((4, 1))
for i in range(4):
    _, rmsef[i], _= reprojection_error(P_estf[i, :, :], px2_vis, pts3d_vis)
    _, rmsee[i], _= reprojection_error(P_este[i, :, :], px2_vis, pts3d_vis)
    print(f"Pose {i} RMSE_F: {rmsef[i, 0]:.2e} px")
    print(f"Pose {i} RMSE_E: {rmsee[i, 0]:.2e} px")

#9 Plot the reprojected points vs true image plane points
min_idx = np.argmin(rmsee)
plot_points(px2_vis, pts3d_vis, P_este[min_idx,:,:], title="True vs. Reprojected Points - Estimate E")
min_idx = np.argmin(rmsef)
plot_points(px2_vis, pts3d_vis, P_estf[min_idx,:,:], title="True vs. Reprojected Points - Estimate F")

sigmas  = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
results = []
results_e = []

rng_noise = np.random.default_rng(7)
for sigma in sigmas:
    px_noisy    = px2_vis + rng_noise.normal(0, sigma, px2_vis.shape)
    points1 = np.vstack((px1_vis, np.ones((1, px1_vis.shape[1]))))
    points2 = np.vstack((px_noisy, np.ones((1, px_noisy.shape[1]))))

    F_n = eight_point(points1.T, points2.T) # Calculate F first then deduce E
    tf, R_1f, R_2f = get_R_t_from_epipolar(F_n, K = K) 
    E_n = eight_point(points1.T, points2.T, K1 = K, K2 = K) # Calculate E directly
    te, R_1e, R_2e = get_R_t_from_epipolar(E_n, K = None)
    
    P_estf_n = P_estimation(tf, R_1f, R_2f, K, norm_t2) # First method
    P_este_n = P_estimation(te, R_1e, R_2e, K, norm_t2) # Second method

    for i in range(4):
        _, rmsef[i], _= reprojection_error(P_estf_n[i, :, :], px2_vis, pts3d_vis)
        _, rmsee[i], _= reprojection_error(P_este_n[i, :, :], px2_vis, pts3d_vis)
    min_idx = np.argmin(rmsef)
    rmse_n = rmsef[min_idx].item()
    min_idx = np.argmin(rmsee)
    rmsee_n = rmsee[min_idx].item()

    results.append(rmse_n)
    results_e.append(rmsee_n)
# ── 5. Visualization
print(results)
print(results_e)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("8points algorithm", fontsize=13, fontweight='bold')


axes[0].plot(sigmas, results, 'o-', color='steelblue', markersize=6)
axes[0].set_xlabel("Bruit pixel (px)")
axes[0].set_ylabel("RMSE reprojection (px)")
axes[0].set_title("Reprojection vs. bruit")
axes[0].grid(True, alpha=0.3)

axes[1].plot(sigmas, results_e, 'o-', color='steelblue', markersize=6)
axes[1].set_xlabel("Bruit pixel (px)")
axes[1].set_ylabel("RMSE reprojection (px)")
axes[1].set_title("Reprojection vs. bruit (E)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


    