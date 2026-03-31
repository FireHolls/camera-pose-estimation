import numpy as np
import matplotlib.pyplot as plt

from simulation.scene_generator import PointsGenerator
from simulation.camera_model import get_K, get_camera_pose
from simulation.projection import project_points, filter_visible
from simulation.dlt import dlt, extract_Rt_from_P
from simulation.dlt_verification import reprojection_error, test_noise

# ── 1. Scene et camera
bounds = np.array([-2.0, 2.0, -1.5, 1.5, 4.0, 10.0])
pts3d  = PointsGenerator(nbPoints=200, seed=42, bounds=bounds)

K_true         = get_K()
R_true, t_true = get_camera_pose(ry=8, tx=0.4, ty=0.0, tz=0.0)

px, depths = project_points(pts3d, K_true, R_true, t_true)
vis        = filter_visible(px, depths)
pts3d_vis  = pts3d[:, vis]
px_vis     = px[:, vis]

# ── 2. DLT + extraction R, t 
P_est        = dlt(px_vis, pts3d_vis)
R_est, t_est = extract_Rt_from_P(P_est, K_true)

_, rmse, _ = reprojection_error(P_est, px_vis, pts3d_vis)
angle_err  = np.degrees(np.arccos(np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)))
t_err      = np.linalg.norm(t_est - t_true)

print(f"RMSE reprojection : {rmse:.2e} px")
print(f"Erreur R          : {angle_err:.4f} deg")
print(f"Erreur t          : {t_err:.2e}")

# ── 3. Robustesse au bruit
sigmas        = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
noise_results = test_noise(px_vis, pts3d_vis, sigmas, seed=7,
                           K=K_true, R_true=R_true, t_true=t_true)

# ── 4. Visualisation 
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("DLT + extraction R, t — Robustesse au bruit", fontsize=13, fontweight='bold')

s_vals = [r[0] for r in noise_results]

axes[0].plot(s_vals, [r[1] for r in noise_results], 'o-', color='steelblue', markersize=6)
axes[0].set_xlabel("Bruit pixel (px)")
axes[0].set_ylabel("RMSE reprojection (px)")
axes[0].set_title("Reprojection vs. bruit")
axes[0].grid(True, alpha=0.3)

axes[1].plot(s_vals, [r[2] for r in noise_results], 'o-', color='tomato', markersize=6)
axes[1].set_xlabel("Bruit pixel (px)")
axes[1].set_ylabel("Erreur angulaire (deg)")
axes[1].set_title("Erreur sur R vs. bruit")
axes[1].grid(True, alpha=0.3)

axes[2].plot(s_vals, [r[3] for r in noise_results], 'o-', color='seagreen', markersize=6)
axes[2].set_xlabel("Bruit pixel (px)")
axes[2].set_ylabel("Erreur ||t_est - t_true||")
axes[2].set_title("Erreur sur t vs. bruit")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
