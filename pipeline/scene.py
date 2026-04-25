import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulation.camera_model import get_K, get_camera_pose
from simulation.projection import project_points, filter_visible


def make_scene(N=100, planar=True, seed=42):
    """
    Generates a simulated scene observed by two cameras.

    Parameters
    ----------
    N      : number of 3D points
    planar : True  → all points on the plane Z=5 (favours homography)
             False → points at random depths Z ∈ [3, 7] (favours fundamental matrix)
    seed   : random seed for reproducibility

    Returns  (dict)
    -------
    pts3d  : (3, M)  visible 3D world points
    px1    : (2, M)  pixels in camera 1
    px2    : (2, M)  pixels in camera 2
    K      : (3, 3)  shared intrinsic matrix
    R_rel  : (3, 3)  ground-truth relative rotation  R2 @ R1ᵀ
    t_rel  : (3,)    ground-truth relative translation
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-2.0, 2.0, N)
    ys = rng.uniform(-1.5, 1.5, N)
    zs = np.full(N, 5.0) if planar else rng.uniform(3.0, 7.0, N)
    pts3d = np.vstack([xs, ys, zs])            # (3, N)

    K = get_K()
    R1, t1 = get_camera_pose()                 # camera 1 : identity
    R2, t2 = get_camera_pose(ry=8, tx=0.4)    # camera 2 : rotated + translated

    R_rel = R2 @ R1.T                          # = R2
    t_rel = t2 - R_rel @ t1                    # = t2 = (0.4, 0, 0)

    px1, d1 = project_points(pts3d, K, R1, t1)
    px2, d2 = project_points(pts3d, K, R2, t2)
    vis = filter_visible(px1, d1) & filter_visible(px2, d2)

    return {
        'pts3d': pts3d[:, vis],
        'px1':   px1[:, vis],
        'px2':   px2[:, vis],
        'K':     K,
        'R_rel': R_rel,
        't_rel': t_rel,
    }
