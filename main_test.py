import numpy as np
import matplotlib.pyplot as plt
from simulation.camera_model import get_K, get_camera_pose
from simulation.scene_generator import PointsGenerator
from simulation.projection import project_points, filter_visible

# 1 Génération de la scène 
bounds = np.array([-5, 5, -5, 5, 3, 8])
pts3d = PointsGenerator(nbPoints=80, seed=42, bounds=bounds)  

# 2 Définition des caméras 
K = get_K()
R1, t1 = np.eye(3), np.zeros(3)
R2, t2 = get_camera_pose(rz=10, tx=1.0)

# 3 Projection 
px1, d1 = project_points(pts3d, K, R1, t1)
px2, d2 = project_points(pts3d, K, R2, t2)

# 4 Filtrage 
vis = filter_visible(px1, d1) & filter_visible(px2, d2)
print(f"Points visibles dans les deux caméras : {vis.sum()} / {pts3d.shape[1]}")

px1_vis = px1[:, vis]
px2_vis = px2[:, vis]

# 5 Visualisation 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Simulation — Projection de points 3D dans deux caméras", fontsize=13)

axes[0].scatter(px1_vis[0], px1_vis[1], s=15, c='steelblue')
axes[0].set_title("Caméra 1")

axes[1].scatter(px2_vis[0], px2_vis[1], s=15, c='tomato')
axes[1].set_title("Caméra 2")

for ax in axes:
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")

plt.tight_layout()
plt.show()