"""
Logique centrale du pipeline : génération de scène, estimation H/F, scoring.
"""

import sys, os, io, contextlib, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulation.camera_model      import get_camera_pose
from simulation.projection        import project_points, filter_visible
from simulation.homography        import homography, decompose_H
from eight_points.eight_point_agl import eight_point
from eight_points.Retrieve_P      import get_R_t_from_epipolar, P_estimation
from score                        import score_H, score_F

H_RATIO_THRESH = 0.45


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self):
        # Scène
        self.scene_type    = 'planar'   # 'planar' | 'nonplanar' | 'custom'
        self.z_min         = 5.0
        self.z_max         = 7.0
        self.x_range       = 2.0
        self.y_range       = 1.5
        self.n_points      = 100
        self.noise_sigma   = 0.0
        self.outlier_ratio = 0.0
        self.seed          = 42

        # Caméra 1 (référence)
        self.cam1_rx = 0.0;  self.cam1_ry = 0.0;  self.cam1_rz = 0.0
        self.cam1_tx = 0.0;  self.cam1_ty = 0.0;  self.cam1_tz = 0.0

        # Caméra 2
        self.cam2_rx = 0.0;  self.cam2_ry = 8.0;  self.cam2_rz = 0.0
        self.cam2_tx = 0.4;  self.cam2_ty = 0.0;  self.cam2_tz = 0.0

        # Intrinsèques
        self.fx    = 1000.0;  self.fy = 1000.0
        self.cx    =  960.0;  self.cy =  540.0
        self.img_w = 1920;    self.img_h = 1080

        # Méthodes
        self.use_H = True
        self.use_F = True

        # Analyse
        self.mode         = 'single'   # 'single' | 'noise_sweep'
        self.noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        self.export_png   = False


# ══════════════════════════════════════════════════════════════════════════════
#  Génération de scène
# ══════════════════════════════════════════════════════════════════════════════

def make_scene(cfg, seed_override=None):
    """
    Génère une scène synthétique vue par deux caméras.
    Retourne un dict : pts3d, px1, px2, K, R_rel, t_rel, R1, t1.
    """
    seed = cfg.seed if seed_override is None else seed_override
    rng  = np.random.default_rng(seed)

    xs = rng.uniform(-cfg.x_range, cfg.x_range, cfg.n_points)
    ys = rng.uniform(-cfg.y_range, cfg.y_range, cfg.n_points)

    if cfg.scene_type == 'planar':
        zs = np.full(cfg.n_points, cfg.z_min)
    elif cfg.scene_type == 'nonplanar':
        lo, hi = min(cfg.z_min, cfg.z_max), max(cfg.z_min, cfg.z_max)
        zs = rng.uniform(lo, hi, cfg.n_points)
    else:
        lo, hi = min(cfg.z_min, cfg.z_max), max(cfg.z_min, cfg.z_max)
        zs = np.full(cfg.n_points, lo) if abs(lo - hi) < 1e-6 else rng.uniform(lo, hi, cfg.n_points)

    pts3d = np.vstack([xs, ys, zs])
    K = np.array([[cfg.fx, 0, cfg.cx],
                  [0, cfg.fy, cfg.cy],
                  [0,      0,     1]], dtype=np.float64)

    R1, t1 = get_camera_pose(cfg.cam1_rx, cfg.cam1_ry, cfg.cam1_rz,
                              cfg.cam1_tx, cfg.cam1_ty, cfg.cam1_tz)
    R2, t2 = get_camera_pose(cfg.cam2_rx, cfg.cam2_ry, cfg.cam2_rz,
                              cfg.cam2_tx, cfg.cam2_ty, cfg.cam2_tz)

    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    px1_c, d1 = project_points(pts3d, K, R1, t1)
    px2_c, d2 = project_points(pts3d, K, R2, t2)
    vis = (filter_visible(px1_c, d1, cfg.img_w, cfg.img_h) &
           filter_visible(px2_c, d2, cfg.img_w, cfg.img_h))

    px1 = px1_c[:, vis].copy()
    px2 = px2_c[:, vis].copy()
    M   = px1.shape[1]

    if cfg.noise_sigma > 0 and M > 0:
        nrng = np.random.default_rng(seed + 1000)
        px1 += nrng.normal(0, cfg.noise_sigma, px1.shape)
        px2 += nrng.normal(0, cfg.noise_sigma, px2.shape)

    if cfg.outlier_ratio > 0 and M > 0:
        orng  = np.random.default_rng(seed + 2000)
        n_out = min(max(1, int(round(M * cfg.outlier_ratio))), M)
        idx   = orng.choice(M, n_out, replace=False)
        px2[0, idx] = orng.uniform(0, cfg.img_w, n_out)
        px2[1, idx] = orng.uniform(0, cfg.img_h, n_out)

    return {
        'pts3d': pts3d[:, vis], 'px1': px1, 'px2': px2,
        'K': K, 'R_rel': R_rel, 't_rel': t_rel,
        'R1': R1, 't1': t1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline H vs F
# ══════════════════════════════════════════════════════════════════════════════

def _rot_err(R_est, R_ref):
    tr = np.trace(R_est.T @ R_ref)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def _trans_err(t_est, t_ref):
    u = t_est / (np.linalg.norm(t_est) + 1e-12)
    v = t_ref / (np.linalg.norm(t_ref) + 1e-12)
    return np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), 0, 1)))


def _best_P(Ps, pts3d, px2, K):
    """Parmi les 4 candidats (R,t) de la décomposition de E, retourne celui avec le RMSE minimal."""
    K_inv   = np.linalg.inv(K)
    pts3d_h = np.vstack([pts3d, np.ones((1, pts3d.shape[1]))])
    best_rmse, best_R, best_t = np.inf, None, None

    for P in Ps:
        proj   = P @ pts3d_h
        depths = proj[2]
        if (depths > 0).mean() < 0.5:
            continue
        mask   = depths > 0
        px_est = proj[:2, mask] / depths[mask]
        rmse   = np.sqrt(np.mean(np.sum((px_est - px2[:, mask]) ** 2, axis=0)))
        if rmse < best_rmse:
            best_rmse = rmse
            Rt = K_inv @ P
            R  = Rt[:, :3]
            U, _, Vt = np.linalg.svd(R)
            R_ = U @ Vt
            if np.linalg.det(R_) < 0:
                Vt[-1] *= -1
                R_ = U @ Vt
            best_R, best_t = R_, Rt[:, 3]

    return best_R, best_t


def run_pipeline(scene, cfg):
    """
    Lance les méthodes activées (H et/ou F) sur une scène.
    Retourne un dict avec scores, poses estimées et erreurs angulaires.
    """
    pts3d, px1, px2, K = scene['pts3d'], scene['px1'], scene['px2'], scene['K']
    R_true, t_true      = scene['R_rel'], scene['t_rel']
    M = px1.shape[1]

    res = dict(
        S_H=None, S_F=None, ratio=None, winner=None,
        R_H=None, t_H=None, R_F=None, t_F=None,
        err_R_H=np.nan, err_t_H=np.nan,
        err_R_F=np.nan, err_t_F=np.nan,
        n_visible=M,
    )

    if M < 4:
        return res

    planar     = (cfg.scene_type == 'planar')
    plane_dist = cfg.z_min if planar else None

    if cfg.use_H and M >= 4:
        try:
            H        = homography(px1, px2)
            S_H      = score_H(H, px1, px2)
            R_H, t_H = decompose_H(H, K, plane_dist=plane_dist, X_ref=pts3d[:, 0])
            res.update(S_H=S_H, R_H=R_H, t_H=t_H)
            if R_H is not None:
                res['err_R_H'] = _rot_err(R_H, R_true)
                res['err_t_H'] = _trans_err(t_H, t_true)
        except Exception:
            pass

    if cfg.use_F and M >= 8:
        try:
            pts1h = np.vstack([px1, np.ones((1, M))]).T
            pts2h = np.vstack([px2, np.ones((1, M))]).T
            F     = eight_point(pts1h, pts2h)
            S_F   = score_F(F, px1, px2)
            with contextlib.redirect_stdout(io.StringIO()):
                t_col, R1_f, R2_f = get_R_t_from_epipolar(F, K)
            Ps       = P_estimation(t_col, R1_f, R2_f, K, s=1)
            R_F, t_F = _best_P(Ps, pts3d, px2, K)
            res.update(S_F=S_F, R_F=R_F, t_F=t_F)
            if R_F is not None:
                res['err_R_F'] = _rot_err(R_F, R_true)
                res['err_t_F'] = _trans_err(t_F, t_true)
        except Exception:
            pass

    sh, sf = res['S_H'], res['S_F']
    if sh is not None and sf is not None:
        ratio = sh / (sh + sf)
        res.update(ratio=ratio, winner='H' if ratio > H_RATIO_THRESH else 'F')
    elif sh is not None:
        res.update(ratio=1.0, winner='H')
    elif sf is not None:
        res.update(ratio=0.0, winner='F')

    return res


def run_noise_sweep(cfg):
    """Lance le pipeline sur chaque niveau de bruit. Retourne une liste de dicts avec clé 'sigma'."""
    levels = sorted(cfg.noise_levels)
    tmp    = copy.copy(cfg)
    results = []
    for sigma in levels:
        tmp.noise_sigma = sigma
        scene = make_scene(tmp)
        res   = run_pipeline(scene, tmp)
        results.append({'sigma': sigma, **res})
    return results
