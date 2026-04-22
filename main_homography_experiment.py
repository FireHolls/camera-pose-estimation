import numpy as np
import matplotlib.pyplot as plt

from simulation.camera_model import get_K, get_camera_pose
from simulation.projection import project_points, filter_visible
from simulation.homography import homography, decompose_H, reprojection_error_H
from plot_fct import plot_points

# ── 1. Planar scene on the plane Z = 0 (standard homography convention)
#    H = K [R[:,0] | R[:,1] | t]  ←→  source (X,Y), Z_world = 0
rng    = np.random.default_rng(42)
N      = 50
xs     = rng.uniform(-2.0,  2.0, N)
ys     = rng.uniform(-1.5,  1.5, N)
pts3d  = np.vstack([xs, ys, np.zeros(N)])   # (3, N) — tous sur Z=0

# Camera offset by tz=8 to observe the Z=0 plane from a positive depth
K_true         = get_K()
R_true, t_true = get_camera_pose(ry=8, tx=0.4, tz=8)

px, depths = project_points(pts3d, K_true, R_true, t_true)
vis        = filter_visible(px, depths)
pts3d_vis  = pts3d[:, vis]
px_vis     = px[:, vis]

print(f"Points visibles : {vis.sum()} / {N}")

# ── 2. Plane coordinates: (X, Y) since Z=0 → standard convention
pts_plane = pts3d_vis[:2, :]    # (2, N)

H_est = homography(pts_plane, px_vis)

_, rmse = reprojection_error_H(H_est, pts_plane, px_vis)
print(f"RMSE reprojection H : {rmse:.2e} px")

# ── 3. Decompose H → R, t  (exact since Z_plane = 0)
X_ref         = pts3d_vis[:, 0]
R_est, t_est  = decompose_H(H_est, K_true, X_ref=X_ref)

angle_err = np.degrees(np.arccos(np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)))
t_err     = np.linalg.norm(t_est - t_true)

print(f"Erreur R            : {angle_err:.4f} deg")
print(f"Erreur t            : {t_err:.2e}")

# ── Visualization: rebuild P from R_est, t_est to use plot_points
P_from_H = K_true @ np.hstack([R_est, t_est.reshape(3, 1)])
plot_points(px_vis, pts3d_vis, P_from_H, title="Homographie — True vs Reprojected (sans bruit)")

# ── 4. Noise robustness
sigmas  = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
results = []

rng_noise = np.random.default_rng(7)
for sigma in sigmas:
    px_noisy    = px_vis + rng_noise.normal(0, sigma, px_vis.shape)

    H_n         = homography(pts_plane, px_noisy)
    _, rmse_n   = reprojection_error_H(H_n, pts_plane, px_noisy)
    R_n, t_n    = decompose_H(H_n, K_true, X_ref=X_ref)

    ang = np.degrees(np.arccos(np.clip((np.trace(R_n.T @ R_true) - 1) / 2, -1, 1)))
    te  = np.linalg.norm(t_n - t_true)
    results.append((sigma, rmse_n, ang, te))

# ── 5. Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Homography H — Noise robustness (planar scene Z=0)", fontsize=13, fontweight='bold')

s_vals = [r[0] for r in results]

axes[0].plot(s_vals, [r[1] for r in results], 'o-', color='steelblue', markersize=6)
axes[0].set_xlabel("Bruit pixel (px)")
axes[0].set_ylabel("RMSE reprojection (px)")
axes[0].set_title("Reprojection vs. bruit")
axes[0].grid(True, alpha=0.3)

axes[1].plot(s_vals, [r[2] for r in results], 'o-', color='tomato', markersize=6)
axes[1].set_xlabel("Bruit pixel (px)")
axes[1].set_ylabel("Erreur angulaire (deg)")
axes[1].set_title("Erreur sur R vs. bruit")
axes[1].grid(True, alpha=0.3)

axes[2].plot(s_vals, [r[3] for r in results], 'o-', color='seagreen', markersize=6)
axes[2].set_xlabel("Bruit pixel (px)")
axes[2].set_ylabel("Erreur ||t_est - t_true||")
axes[2].set_title("Erreur sur t vs. bruit")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
