import numpy as np
import matplotlib.pyplot as plt

from simulation.camera_model import get_K, get_camera_pose
from simulation.projection import project_points, filter_visible
from simulation.homography import homography, decompose_H, reprojection_error_H
from plot_fct import plot_points

# ── 1. Planar scene at Z=5 in world frame
#    Camera 1 at identity (R=I, t=0): Z_cam1 = Z_world = 5 > 0 ✓
rng   = np.random.default_rng(42)
N     = 50
xs    = rng.uniform(-2.0,  2.0, N)
ys    = rng.uniform(-1.5,  1.5, N)
pts3d = np.vstack([xs, ys, np.full(N, 5.0)])   # (3, N)

K = get_K()

# Camera 1: reference frame (identity)
R1, t1 = get_camera_pose()                   # R=I, t=0

# Camera 2: rotated and translated relative to camera 1
R2, t2 = get_camera_pose(ry=8, tx=0.4)      # R=Ry(8°), t=(0.4, 0, 0)

# Ground truth relative pose (cam1 → cam2)
R_rel_true = R2 @ R1.T                       # = R2
t_rel_true = t2 - R_rel_true @ t1            # = t2 = (0.4, 0, 0)

PLANE_DIST = 5.0   # depth of the Z=5 plane in camera 1's frame

# ── 2. Project into both cameras, keep points visible in both
px1, depths1 = project_points(pts3d, K, R1, t1)
px2, depths2 = project_points(pts3d, K, R2, t2)

vis      = filter_visible(px1, depths1) & filter_visible(px2, depths2)
pts3d_v  = pts3d[:, vis]
px1_v    = px1[:, vis]
px2_v    = px2[:, vis]

print(f"Points visibles dans les 2 vues : {vis.sum()} / {N}")

# ── 3. Estimate H: px1 → px2  (two-view homography)
H_est = homography(px1_v, px2_v)

_, rmse = reprojection_error_H(H_est, px1_v, px2_v)
print(f"RMSE reprojection H : {rmse:.2e} px")

# ── 4. Decompose H → R_rel, t_rel
X_ref         = pts3d_v[:, 0]
R_est, t_est  = decompose_H(H_est, K, plane_dist=PLANE_DIST, X_ref=X_ref)

angle_err = np.degrees(np.arccos(np.clip((np.trace(R_est.T @ R_rel_true) - 1) / 2, -1, 1)))
t_err     = np.linalg.norm(t_est - t_rel_true)

print(f"Erreur R            : {angle_err:.4f} deg")
print(f"Erreur t            : {t_err:.2e}")

# ── Visualize: use P2_est = K[R_est|t_est] to reproject pts3d into camera 2
P2_est = K @ np.hstack([R_est, t_est.reshape(3, 1)])
plot_points(px2_v, pts3d_v, P2_est, title="Homographie 2 vues — True vs Reprojected (sans bruit)")

# ── 5. Noise robustness
sigmas  = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
results = []

rng_noise = np.random.default_rng(7)
for sigma in sigmas:
    px1_noisy = px1_v + rng_noise.normal(0, sigma, px1_v.shape)
    px2_noisy = px2_v + rng_noise.normal(0, sigma, px2_v.shape)

    H_n        = homography(px1_noisy, px2_noisy)
    _, rmse_n  = reprojection_error_H(H_n, px1_noisy, px2_noisy)
    R_n, t_n   = decompose_H(H_n, K, plane_dist=PLANE_DIST, X_ref=X_ref)

    ang = np.degrees(np.arccos(np.clip((np.trace(R_n.T @ R_rel_true) - 1) / 2, -1, 1)))
    te  = np.linalg.norm(t_n - t_rel_true)
    results.append((sigma, rmse_n, ang, te))

# ── 6. Plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Homography (2-view, planar scene Z=5) — Noise robustness", fontsize=13, fontweight='bold')

s_vals = [r[0] for r in results]

axes[0].plot(s_vals, [r[1] for r in results], 'o-', color='steelblue', markersize=6)
axes[0].set_xlabel("Pixel noise σ (px)")
axes[0].set_ylabel("RMSE reprojection (px)")
axes[0].set_title("Reprojection vs. noise")
axes[0].grid(True, alpha=0.3)

axes[1].plot(s_vals, [r[2] for r in results], 'o-', color='tomato', markersize=6)
axes[1].set_xlabel("Pixel noise σ (px)")
axes[1].set_ylabel("Angular error (deg)")
axes[1].set_title("Rotation error vs. noise")
axes[1].grid(True, alpha=0.3)

axes[2].plot(s_vals, [r[3] for r in results], 'o-', color='seagreen', markersize=6)
axes[2].set_xlabel("Pixel noise σ (px)")
axes[2].set_ylabel("‖t_est − t_true‖")
axes[2].set_title("Translation error vs. noise")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
