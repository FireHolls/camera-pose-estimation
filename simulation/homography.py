import numpy as np


def _build_A_homography(pts_src, pts_dst):
    """
    Builds the matrix A of size (2N × 9) for the homogeneous system A h = 0.

    For each correspondence (x,y) -> (u,v), the constraint x_dst × (H x_src) = 0
    yields 2 independent equations:

      Row 2i   : [-x, -y, -1,  0,  0,  0,  u·x,  u·y,  u]
      Row 2i+1 : [ 0,  0,  0, -x, -y, -1,  v·x,  v·y,  v]

    inputs
    pts_src : (2, N) ndarray — source coordinates (3D plane or image 1)
    pts_dst : (2, N) ndarray — destination coordinates (image)

    Returns
    A : (2N, 9) ndarray
    """
    N = pts_src.shape[1]
    A = np.zeros((2 * N, 9), dtype=np.float64)

    for i in range(N):
        x, y = pts_src[:, i]
        u, v = pts_dst[:, i]

        A[2 * i,     :] = [-x, -y, -1,   0,   0,  0,  u*x,  u*y,  u]
        A[2 * i + 1, :] = [  0,  0,  0,  -x,  -y, -1,  v*x,  v*y,  v]

    return A


def homography(pts_src, pts_dst, normalize=True):
    """
    Estimates the 3×3 homography matrix H from N ≥ 4 2D↔2D correspondences.

    Implementation is identical to DLT but H is 3×3 (8 DOF) instead of P 3×4 (11 DOF)
    → 4 correspondences are sufficient (instead of 6).

    pts_src   : (2, N) ndarray
    pts_dst   : (2, N) ndarray
    normalize : apply normalization

    Returns
    H : (3, 3) ndarray — estimated homography, normalized ‖H‖_F = 1
    """
    N = pts_src.shape[1]
    assert N >= 4, (
        f"Homography requires at least 4 correspondences, got {N}. "
        "H has 8 DOF; each correspondence gives 2 equations → ⌈8/2⌉ = 4 minimum."
    )

    if normalize:
        pts_src_n, T_src = _normalize_2d(pts_src)
        pts_dst_n, T_dst = _normalize_2d(pts_dst)
    else:
        pts_src_n, pts_dst_n = pts_src, pts_dst

    A = _build_A_homography(pts_src_n, pts_dst_n)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]               # (9,) — vecteur solution
    H_norm = h.reshape(3, 3)

    if normalize:
        # Denormalization : H_true = T_dst⁻¹ @ H_norm @ T_src
        H = np.linalg.inv(T_dst) @ H_norm @ T_src
    else:
        H = H_norm

    H /= np.linalg.norm(H)

    return H


def _normalize_2d(points_2d):
    """
    Isotropic normalization for 2D coordinates.
    -> centering at centroid + mean distance = sqrt(2).

    points_2d : (2, N) ndarray

    Returns
    pts_norm : (2, N) ndarray
    T        : (3, 3) ndarray — normalization matrix
    """
    N = points_2d.shape[1]

    cx = points_2d[0].mean()
    cy = points_2d[1].mean()
    shifted = points_2d - np.array([[cx], [cy]])

    mean_dist = np.mean(np.sqrt(shifted[0] ** 2 + shifted[1] ** 2))
    s = np.sqrt(2) / mean_dist

    T = np.array([
        [s, 0, -s * cx],
        [0, s, -s * cy],
        [0, 0,       1]
    ], dtype=np.float64)

    pts_h = np.vstack([points_2d, np.ones((1, N))])
    pts_norm = T @ pts_h

    return pts_norm[:2], T


def decompose_H(H, K1, K2=None, plane_dist=None, X_ref=None):
    """
    Extracts relative rotation R and translation t between two cameras from
    a homography H estimated between their two image planes (pixel → pixel).

    For a planar scene at depth d in camera 1's frame, with plane normal n = (0,0,1):
      H12 = K2 (R_rel + t_rel nᵀ / d) K1⁻¹
      B = K2⁻¹ H12 K1 = λ [R_rel[:,0] | R_rel[:,1] | R_rel[:,2] + t_rel/d]

    r1 = B[:,0]/λ,  r2 = B[:,1]/λ,  r3 = r1 × r2
    R_rel = projection of [r1|r2|r3] onto SO(3) via SVD
    t_rel/d = B[:,2]/λ − R_rel[:,2]

    H          : (3, 3) — estimated homography (camera1 pixels → camera2 pixels)
    K1         : (3, 3) — intrinsics of camera 1
    K2         : (3, 3) — intrinsics of camera 2 (default: same as K1)
    plane_dist : float  — depth of the plane in camera 1's frame (= distance from
                          cam1 to the plane). If given, returns absolute t_rel.
                          If None, returns unit translation direction.
    X_ref      : (3,)   — world reference point for chirality check (must be visible
                          in both cameras, i.e. positive depth in both frames)

    Returns
    R : (3, 3) — relative rotation  R_rel = R2 @ R1ᵀ
    t : (3,)   — relative translation (absolute if plane_dist given, unit vector otherwise)
    """
    if K2 is None:
        K2 = K1

    B = np.linalg.inv(K2) @ H @ K1   # λ [r1 | r2 | r3 + t_rel/d]

    lam = (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1])) / 2

    def _from_lambda(sign):
        l = sign * lam
        r1 = B[:, 0] / l
        r2 = B[:, 1] / l
        r3 = np.cross(r1, r2)
        R_approx = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(R_approx)
        R_ = U @ Vt
        if np.linalg.det(R_) < 0:
            Vt[-1, :] *= -1
            R_ = U @ Vt
        t_dir = B[:, 2] / l - R_[:, 2]          # = t_rel / d  (up to scale)
        if plane_dist is not None:
            t_ = plane_dist * t_dir
        else:
            n = np.linalg.norm(t_dir)
            t_ = t_dir / n if n > 1e-10 else t_dir
        return R_, t_

    R, t = _from_lambda(+1)

    # H is defined up to ±1 — pick the sign such that X_ref is in front of camera 2
    if X_ref is not None:
        if (R @ X_ref + t)[2] < 0:
            R, t = _from_lambda(-1)

    return R, t


def reprojection_error_H(H, pts_src, pts_dst):
    """
    Computes the reprojection error of homography H.

    pts_src : (2, N) ndarray
    pts_dst : (2, N) ndarray — observed pixels

    Returns
    errors : (N,)  ndarray — per-point Euclidean error (pixels)
    rmse   : float         — root mean square error
    """
    N = pts_src.shape[1]
    pts_src_h = np.vstack([pts_src, np.ones((1, N))])  # (3, N)

    projected = H @ pts_src_h                           # (3, N)
    projected /= projected[2, :]                        # normalisation homogène

    diff = projected[:2, :] - pts_dst                   # (2, N)
    errors = np.sqrt((diff ** 2).sum(axis=0))            # (N,)
    rmse = np.sqrt(np.mean(errors ** 2))

    return errors, rmse
