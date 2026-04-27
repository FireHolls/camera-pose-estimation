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

# 4 Filtrage 
vis = filter_visible(px1, d1) & filter_visible(px2, d2)
pts3d_vis = pts3d[:, vis]
print(f"Points visibles dans les deux caméras : {vis.sum()} / {pts3d.shape[1]}")
px1_vis = px1[:, vis]
px2_vis = px2[:, vis]

#5 Turn the points to homogenous 2D points

#6 Retrieve the translation vector and rotation matrices
F = eight_point(px1_vis, px2_vis) # Calculate F first then deduce E
tf, R_1f, R_2f = get_R_t_from_epipolar(F, K = K) 

#7 Estrimate the projection matrix
P_estf = P_estimation(tf, R_1f, R_2f, K) # First method
R2_hat, t2_norm, P2_norm = parallax(P_estf, K, px1_vis, px2_vis)
s = find_scaling_factor(P2_norm, K, px1_vis, px2_vis, pts3d_vis)
t2_hat = s*t2_norm
P2 = K @ np.hstack((R2_hat, t2_hat))

P1 = K@np.hstack((np.eye(3), np.zeros((3, 1))))


pts3D_triag = triangulate(P1, P2, px1_vis, px2_vis)
plot_points(px2_vis, pts3D_triag, P2, title="True vs. Reprojected Points - Estimate F")

sigmas  = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
results = []
results_triag = []

rng_noise = np.random.default_rng(7)
P1 = K@np.hstack((np.eye(3), np.zeros((3, 1))))
for sigma in sigmas:
    px_noisy    = px2_vis + rng_noise.normal(0, sigma, px2_vis.shape)

    F_n = eight_point(px1_vis, px_noisy) # Calculate F first then deduce E
    tf, R_1f, R_2f = get_R_t_from_epipolar(F_n, K = K) 
    
    P_estf_n = P_estimation(tf, R_1f, R_2f, K) # First method
    R2_hat, t2_norm, P2_norm = parallax(P_estf_n, K, px1_vis, px_noisy)
    s = find_scaling_factor(P2_norm, K, px1_vis, px_noisy, pts3d_vis)
    print(s)
    t2_hat = s*t2_norm
    P2 = K @ np.hstack((R2_hat, t2_hat))
    print(t2_hat)
    _, rmsef, _= reprojection_error(P2, px2_vis, pts3d_vis)
    _, rmsef_triag, _= reprojection_error(P2, px2_vis, pts3D_triag)

    results.append(rmsef)
    results_triag.append(rmsef_triag)


# ── 5. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("8points algorithm + 3D ground truth", fontsize=13, fontweight='bold')


axes[0].plot(sigmas, results, 'o-', color='steelblue', markersize=6)
axes[0].set_xlabel("Bruit pixel (px)")
axes[0].set_ylabel("RMSE reprojection (px)")
axes[0].set_title("Reprojection vs. bruit")
axes[0].grid(True, alpha=0.3)

axes[1].plot(sigmas, results_triag, 'o-', color='steelblue', markersize=6)
axes[1].set_xlabel("Bruit pixel (px)")
axes[1].set_ylabel("RMSE reprojection (px)")
axes[1].set_title("Reprojection vs. bruit (Triag)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


    