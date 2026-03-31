import numpy as np


def reprojection_error(P, points_2d, points_3d):
    """
    Calcule l'erreur de reprojection par point ||x_i - pi(P, X_i)||_2.

    La reprojection de X_i par P :
      [u, v, w]^T = P @ [X, Y, Z, 1]^T
      u_est = u/w,  v_est = v/w

    Paramètres
    P          : (3, 4) ndarray
    points_2d  : (2, N) ndarray — obs pixel (truth)
    points_3d  : (3, N) ndarray — points 3D monde

    Retourne
    errors  : (N,) ndarray — erreur par point en pixels
    rmse    : float
    max_err : float
    """
    N = points_3d.shape[1]
    X_h = np.vstack([points_3d, np.ones((1, N))])
    proj = P @ X_h
    u_est = proj[0] / proj[2]
    v_est = proj[1] / proj[2]
    errors = np.sqrt((u_est - points_2d[0]) ** 2 + (v_est - points_2d[1]) ** 2)
    return errors, np.sqrt(np.mean(errors ** 2)), errors.max()


def _pose_errors(P_est, K, R_true, t_true):
    """
    Calcule l'erreur angulaire (degrés) sur R et l'erreur euclidienne sur t.
    """
    from simulation.dlt import extract_Rt_from_P
    R_est, t_est = extract_Rt_from_P(P_est, K)
    cos_angle = np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1.0, 1.0)
    angle_err = np.degrees(np.arccos(cos_angle))
    t_err = np.linalg.norm(t_est - t_true)
    return angle_err, t_err


def test_noise(points_2d_clean, points_3d, sigmas, seed=0, K=None, R_true=None, t_true=None):
    """
    Évalue les métriques en fonction du bruit pixel

    Paramètres
    points_2d_clean : (2, N) ndarray
    points_3d       : (3, N) ndarray
    sigmas          : list of float  niveaux de bruit (px)
    K, R_true, t_true : si fournis calcule aussi les erreurs sur R et t

    Retourne
    list of (sigma, rmse)  ou  (sigma, rmse, angle_err, t_err) si K fourni
    """
    from simulation.dlt import dlt

    compute_pose = K is not None
    rng = np.random.default_rng(seed)
    results = []
    for sigma in sigmas:
        P_est = dlt(points_2d_clean + rng.normal(0.0, sigma, size=points_2d_clean.shape),
                    points_3d)
        _, rmse, _ = reprojection_error(P_est, points_2d_clean, points_3d)
        if compute_pose:
            angle_err, t_err = _pose_errors(P_est, K, R_true, t_true)
            results.append((sigma, rmse, angle_err, t_err))
        else:
            results.append((sigma, rmse))
    return results


def test_npoints(points_2d, points_3d, ns, seed=0, K=None, R_true=None, t_true=None):
    """
    Évalue les métriques en fonction du nombre de correspondances utilisées.

    Paramètres
    points_2d : (2, N) ndarray
    points_3d : (3, N) ndarray
    ns        : list of int — nombres de points à utiliser (>= 6)
    K, R_true, t_true : si fournis, calcule aussi les erreurs sur R et t

    Retourne
    list of (n, rmse) ou  (n, rmse, angle_err, t_err) si K fourni
    """
    from simulation.dlt import dlt

    compute_pose = K is not None
    rng = np.random.default_rng(seed)
    total = points_3d.shape[1]
    results = []
    for n in ns:
        if n > total:
            break
        idx = rng.choice(total, size=n, replace=False)
        P_est = dlt(points_2d[:, idx], points_3d[:, idx])
        _, rmse, _ = reprojection_error(P_est, points_2d, points_3d)
        if compute_pose:
            angle_err, t_err = _pose_errors(P_est, K, R_true, t_true)
            results.append((n, rmse, angle_err, t_err))
        else:
            results.append((n, rmse))
    return results
