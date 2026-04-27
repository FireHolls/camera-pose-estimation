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
points1 = np.vstack((px1_vis, np.ones((1, px1_vis.shape[1]))))
points2 = np.vstack((px2_vis, np.ones((1, px2_vis.shape[1]))))