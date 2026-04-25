"""
Camera Pose Estimation — Outil de Recherche Interactif

Lance :  python pipeline/research_tool.py
"""

import sys, os, io, contextlib, copy
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TOOL_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TOOL_DIR)

import numpy as np
import matplotlib.pyplot as plt

from simulation.camera_model      import get_camera_pose
from simulation.projection        import project_points, filter_visible
from simulation.homography        import homography, decompose_H
from eight_points.eight_point_agl import eight_point
from eight_points.Retrieve_P      import get_R_t_from_epipolar, P_estimation
from score                        import score_H, score_F
from visualize3d                  import plot_scene_3d

H_RATIO_THRESH = 0.45


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self):
        # -- Scene -------------------------------------------------------
        self.scene_type    = 'planar'   # 'planar' | 'nonplanar' | 'custom'
        self.z_min         = 5.0        # plan Z (planar) ou borne basse (custom)
        self.z_max         = 5.0        # borne haute (custom uniquement)
        self.x_range       = 2.0        # points X in [-x_range, +x_range]
        self.y_range       = 1.5        # points Y in [-y_range, +y_range]
        self.n_points      = 100
        self.noise_sigma   = 0.0        # ecart-type bruit pixel (px)
        self.outlier_ratio = 0.0        # fraction de fausses correspondances [0-1]
        self.seed          = 42

        # -- Camera 1 (reference) ----------------------------------------
        self.cam1_rx = 0.0;  self.cam1_ry = 0.0;  self.cam1_rz = 0.0
        self.cam1_tx = 0.0;  self.cam1_ty = 0.0;  self.cam1_tz = 0.0

        # -- Camera 2 ----------------------------------------------------
        self.cam2_rx = 0.0;  self.cam2_ry = 8.0;  self.cam2_rz = 0.0
        self.cam2_tx = 0.4;  self.cam2_ty = 0.0;  self.cam2_tz = 0.0

        # -- Intrinseques ------------------------------------------------
        self.fx    = 1000.0;  self.fy = 1000.0
        self.cx    =  960.0;  self.cy =  540.0
        self.img_w = 1920;    self.img_h = 1080

        # -- Methodes ----------------------------------------------------
        self.use_H = True
        self.use_F = True

        # -- Mode d'analyse ----------------------------------------------
        self.mode         = 'single'   # 'single' | 'noise_sweep' | 'montecarlo'
        self.noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        self.mc_n         = 50
        self.export_png   = False


# ══════════════════════════════════════════════════════════════════════════════
#  Generation de scene
# ══════════════════════════════════════════════════════════════════════════════

def make_scene(cfg, seed_override=None):
    seed = cfg.seed if seed_override is None else seed_override
    rng  = np.random.default_rng(seed)

    xs = rng.uniform(-cfg.x_range, cfg.x_range, cfg.n_points)
    ys = rng.uniform(-cfg.y_range, cfg.y_range, cfg.n_points)

    if cfg.scene_type == 'planar':
        zs = np.full(cfg.n_points, cfg.z_min)
    elif cfg.scene_type == 'nonplanar':
        zs = rng.uniform(3.0, 7.0, cfg.n_points)
    else:
        lo, hi = min(cfg.z_min, cfg.z_max), max(cfg.z_min, cfg.z_max)
        if abs(lo - hi) < 1e-6:
            zs = np.full(cfg.n_points, lo)
        else:
            zs = rng.uniform(lo, hi, cfg.n_points)

    pts3d = np.vstack([xs, ys, zs])

    K  = np.array([[cfg.fx, 0, cfg.cx],
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

    pts3d_v = pts3d[:, vis]
    px1     = px1_c[:, vis].copy()
    px2     = px2_c[:, vis].copy()
    M       = px1.shape[1]

    # Bruit pixel
    if cfg.noise_sigma > 0 and M > 0:
        nrng = np.random.default_rng(seed + 1000)
        px1 += nrng.normal(0, cfg.noise_sigma, px1.shape)
        px2 += nrng.normal(0, cfg.noise_sigma, px2.shape)

    # Outliers : on remplace n_out correspondances de px2 par du bruit aleatoire
    if cfg.outlier_ratio > 0 and M > 0:
        orng  = np.random.default_rng(seed + 2000)
        n_out = max(1, int(round(M * cfg.outlier_ratio)))
        n_out = min(n_out, M)
        idx   = orng.choice(M, n_out, replace=False)
        px2[0, idx] = orng.uniform(0, cfg.img_w, n_out)
        px2[1, idx] = orng.uniform(0, cfg.img_h, n_out)

    return {
        'pts3d': pts3d_v, 'px1': px1, 'px2': px2,
        'K': K, 'R_rel': R_rel, 't_rel': t_rel,
        'R1': R1, 't1': t1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions utilitaires algorithmes
# ══════════════════════════════════════════════════════════════════════════════

def _rot_err(R_est, R_ref):
    tr = np.trace(R_est.T @ R_ref)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def _trans_err(t_est, t_ref):
    u = t_est / (np.linalg.norm(t_est) + 1e-12)
    v = t_ref / (np.linalg.norm(t_ref) + 1e-12)
    return np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), 0, 1)))


def _best_P(Ps, pts3d, px2, K):
    K_inv    = np.linalg.inv(K)
    pts3d_h  = np.vstack([pts3d, np.ones((1, pts3d.shape[1]))])
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
            best_R = R_
            best_t = Rt[:, 3]
    return best_R, best_t


def run_pipeline(scene, cfg):
    """Execute les methodes activees sur une scene. Retourne un dict de resultats."""
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

    # -- Homographie ---------------------------------------------------
    if cfg.use_H and M >= 4:
        try:
            H     = homography(px1, px2)
            S_H   = score_H(H, px1, px2)
            R_H, t_H = decompose_H(H, K, plane_dist=plane_dist, X_ref=pts3d[:, 0])
            res.update(S_H=S_H, R_H=R_H, t_H=t_H)
            if R_H is not None:
                res['err_R_H'] = _rot_err(R_H, R_true)
                res['err_t_H'] = _trans_err(t_H, t_true)
        except Exception:
            pass

    # -- Matrice fondamentale (8 points) --------------------------------
    if cfg.use_F and M >= 8:
        try:
            pts1h = np.vstack([px1, np.ones((1, M))]).T
            pts2h = np.vstack([px2, np.ones((1, M))]).T
            F     = eight_point(pts1h, pts2h)
            S_F   = score_F(F, px1, px2)
            with contextlib.redirect_stdout(io.StringIO()):
                t_col, R1_f, R2_f = get_R_t_from_epipolar(F, K)
            Ps   = P_estimation(t_col, R1_f, R2_f, K, s=1)
            R_F, t_F = _best_P(Ps, pts3d, px2, K)
            res.update(S_F=S_F, R_F=R_F, t_F=t_F)
            if R_F is not None:
                res['err_R_F'] = _rot_err(R_F, R_true)
                res['err_t_F'] = _trans_err(t_F, t_true)
        except Exception:
            pass

    # -- Decision ORB-SLAM style ----------------------------------------
    sh, sf = res['S_H'], res['S_F']
    if sh is not None and sf is not None:
        ratio = sh / (sh + sf)
        res.update(ratio=ratio, winner='H' if ratio > H_RATIO_THRESH else 'F')
    elif sh is not None:
        res.update(ratio=1.0, winner='H')
    elif sf is not None:
        res.update(ratio=0.0, winner='F')

    return res


# ══════════════════════════════════════════════════════════════════════════════
#  Modes d'analyse
# ══════════════════════════════════════════════════════════════════════════════

def _print_res(res):
    sep = '  ' + '-' * 52
    print(sep)
    print(f"  Points visibles : {res['n_visible']}")
    if res['S_H'] is not None:
        print(f"  S_H = {res['S_H']:.1f}")
    if res['S_F'] is not None:
        print(f"  S_F = {res['S_F']:.1f}")
    if res['ratio'] is not None:
        print(f"  Ratio = {res['ratio']:.3f}  ->  Vainqueur : [{res['winner']}]")
    print(sep)
    print(f"  {'Methode':8}  {'Err R (deg)':>12}  {'Err t dir (deg)':>16}")
    if res['S_H'] is not None:
        print(f"  {'H':8}  {res['err_R_H']:>12.4f}  {res['err_t_H']:>16.4f}")
    if res['S_F'] is not None:
        print(f"  {'F':8}  {res['err_R_F']:>12.4f}  {res['err_t_F']:>16.4f}")
    print()


def _save_fig(cfg, suffix):
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(TOOL_DIR, f'result_{suffix}_{ts}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Figure sauvegardee : {out}")


# ── Mode 1 : Scene unique ──────────────────────────────────────────────────────

def analyse_single(cfg):
    print("\n  Generation de la scene...", end=' ', flush=True)
    scene = make_scene(cfg)
    M     = scene['pts3d'].shape[1]
    print(f"{M} points visibles.")

    if M < 4:
        print("  Pas assez de points. Modifiez la configuration.")
        return

    res = run_pipeline(scene, cfg)
    _print_res(res)

    if res['winner'] is None:
        print("  Impossible de determiner un vainqueur (points insuffisants).")
        return

    plot_scene_3d(
        scene,
        res['R_H'], res['t_H'],
        res['R_F'], res['t_F'],
        res['winner'],
        res['S_H'] or 0.0,
        res['S_F'] or 0.0,
        res['ratio'] or 0.0,
        H_RATIO_THRESH,
        cfg.scene_type == 'planar',
        img_shape=(cfg.img_w, cfg.img_h),
    )

    if cfg.export_png:
        _save_fig(cfg, 'single')


# ── Mode 2 : Balayage bruit ────────────────────────────────────────────────────

def analyse_noise_sweep(cfg):
    levels = sorted(cfg.noise_levels)
    print(f"\n  Balayage bruit : sigma = {levels} px")
    print(f"  Seed fixe : {cfg.seed}  (meme scene, bruit croissant)\n")

    err_R_H, err_t_H = [], []
    err_R_F, err_t_F = [], []

    tmp = copy.copy(cfg)
    for sigma in levels:
        tmp.noise_sigma = sigma
        scene = make_scene(tmp)
        res   = run_pipeline(scene, tmp)
        err_R_H.append(res['err_R_H'])
        err_t_H.append(res['err_t_H'])
        err_R_F.append(res['err_R_F'])
        err_t_F.append(res['err_t_F'])
        parts = []
        if cfg.use_H: parts.append(f"err_R_H={res['err_R_H']:.3f}")
        if cfg.use_F: parts.append(f"err_R_F={res['err_R_F']:.3f}")
        print(f"    sigma={sigma:.2f} px  |  {' | '.join(parts)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Robustesse au bruit — Erreur angulaire vs sigma',
                 fontsize=12, fontweight='bold')

    for ax, (yH, yF, ylabel, title) in zip(
        [ax1, ax2],
        [(err_R_H, err_R_F, 'Erreur rotation (deg)',     'Rotation'),
         (err_t_H, err_t_F, 'Erreur translation dir (deg)', 'Translation')],
    ):
        if cfg.use_H:
            ax.plot(levels, yH, 'o-', color='#F57C00', lw=2, ms=6, label='Homographie H')
        if cfg.use_F:
            ax.plot(levels, yF, 's-', color='#D32F2F', lw=2, ms=6, label='Matrice F')
        ax.set_xlabel('Bruit sigma (px)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if cfg.export_png:
        _save_fig(cfg, 'noise_sweep')
    plt.show()


# ── Mode 3 : Monte-Carlo ───────────────────────────────────────────────────────

def analyse_montecarlo(cfg):
    n = cfg.mc_n
    print(f"\n  Monte-Carlo : {n} scenes independantes (seed {cfg.seed} a {cfg.seed + n - 1})")

    data = {'R_H': [], 'R_F': [], 't_H': [], 't_F': []}
    wins = {'H': 0, 'F': 0, None: 0}

    for i in range(n):
        scene = make_scene(cfg, seed_override=cfg.seed + i)
        res   = run_pipeline(scene, cfg)
        data['R_H'].append(res['err_R_H'])
        data['R_F'].append(res['err_R_F'])
        data['t_H'].append(res['err_t_H'])
        data['t_F'].append(res['err_t_F'])
        wins[res['winner']] = wins.get(res['winner'], 0) + 1
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n} scenes traitees...")

    print(f"\n  Victoires — H : {wins.get('H',0)}  F : {wins.get('F',0)}")

    def _clean(arr):
        a = np.array(arr, dtype=float)
        return a[~np.isnan(a)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Monte-Carlo ({n} scenes) — Distribution des erreurs angulaires',
                 fontsize=12, fontweight='bold')

    for ax, kH, kF, ylabel, title in zip(
        [ax1, ax2],
        ['R_H', 't_H'], ['R_F', 't_F'],
        ['Erreur rotation (deg)', 'Erreur translation dir (deg)'],
        ['Rotation', 'Translation'],
    ):
        plot_data, labels, colors = [], [], []
        if cfg.use_H:
            d = _clean(data[kH])
            if len(d):
                plot_data.append(d)
                labels.append('H')
                colors.append('#F57C00')
                print(f"  {kH:4} : moy={d.mean():.3f} deg  std={d.std():.3f} deg"
                      f"  med={np.median(d):.3f} deg")
        if cfg.use_F:
            d = _clean(data[kF])
            if len(d):
                plot_data.append(d)
                labels.append('F')
                colors.append('#D32F2F')
                print(f"  {kF:4} : moy={d.mean():.3f} deg  std={d.std():.3f} deg"
                      f"  med={np.median(d):.3f} deg")

        if plot_data:
            bp = ax.boxplot(plot_data, patch_artist=True, widths=0.45,
                            medianprops=dict(color='black', lw=2))
            for patch, col in zip(bp['boxes'], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.72)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if cfg.export_png:
        _save_fig(cfg, 'montecarlo')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers saisie utilisateur
# ══════════════════════════════════════════════════════════════════════════════

def _cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def _ask(prompt, current):
    """Demande une valeur ; Entree conserve la valeur actuelle."""
    val = input(f"    {prompt}  [{current}] : ").strip()
    return val if val else str(current)


def _ask_float(prompt, current, vmin=None, vmax=None):
    while True:
        s = _ask(prompt, f'{current:.4g}')
        try:
            v = float(s)
            if vmin is not None and v < vmin:
                print(f"    Minimum autorise : {vmin}")
                continue
            if vmax is not None and v > vmax:
                print(f"    Maximum autorise : {vmax}")
                continue
            return v
        except ValueError:
            print("    Nombre attendu (ex: 3.14)")


def _ask_int(prompt, current, vmin=None, vmax=None):
    while True:
        s = _ask(prompt, current)
        try:
            v = int(s)
            if vmin is not None and v < vmin:
                print(f"    Minimum autorise : {vmin}")
                continue
            if vmax is not None and v > vmax:
                print(f"    Maximum autorise : {vmax}")
                continue
            return v
        except ValueError:
            print("    Entier attendu (ex: 100)")


def _ask_vec3(label, a, b, c, unit=''):
    """Demande 3 flottants sur une ligne."""
    unit_str = f' ({unit})' if unit else ''
    while True:
        s = _ask(f"{label}{unit_str}  [3 valeurs separees par espaces]",
                 f"{a:.4g} {b:.4g} {c:.4g}")
        parts = s.split()
        if len(parts) == 3:
            try:
                return [float(x) for x in parts]
            except ValueError:
                pass
        print("    3 valeurs numeriques requises  (ex: 0 8 0)")


def _ask_yesno(prompt, current):
    s = _ask(f"{prompt}  (o=oui / n=non)", 'o' if current else 'n').lower()
    return s in ('o', 'oui', 'y', 'yes', '1', 'true')


# ══════════════════════════════════════════════════════════════════════════════
#  Sous-menus d'edition
# ══════════════════════════════════════════════════════════════════════════════

def _pick_scene_type(cfg):
    _cls()
    print("  -- TYPE DE SCENE --\n")
    print("  [1] Planaire       tous les points a Z = constante")
    print("  [2] Non-planaire   Z aleatoire in [3, 7] m")
    print("  [3] Personnalise   plage Z libre\n")
    c = input("  Choix : ").strip()
    if c == '1':
        cfg.scene_type = 'planar'
        cfg.z_min = cfg.z_max = _ask_float("Profondeur Z (m)", cfg.z_min, vmin=0.1)
    elif c == '2':
        cfg.scene_type = 'nonplanar'
    elif c == '3':
        cfg.scene_type = 'custom'
        cfg.z_min = _ask_float("Z minimum (m)", cfg.z_min, vmin=0.1)
        cfg.z_max = _ask_float("Z maximum (m)", max(cfg.z_max, cfg.z_min), vmin=cfg.z_min)


def _edit_xy_range(cfg):
    cfg.x_range = _ask_float("Etendue X  (points de -X a +X, m)", cfg.x_range, vmin=0.01)
    cfg.y_range = _ask_float("Etendue Y  (points de -Y a +Y, m)", cfg.y_range, vmin=0.01)


def _edit_rotation(cfg, cam_id):
    pre = f'cam{cam_id}_'
    rx, ry, rz = getattr(cfg, f'{pre}rx'), getattr(cfg, f'{pre}ry'), getattr(cfg, f'{pre}rz')
    rx, ry, rz = _ask_vec3(f"Rotation camera {cam_id}  rx ry rz", rx, ry, rz, unit='deg')
    setattr(cfg, f'{pre}rx', rx)
    setattr(cfg, f'{pre}ry', ry)
    setattr(cfg, f'{pre}rz', rz)


def _edit_translation(cfg, cam_id):
    pre = f'cam{cam_id}_'
    tx, ty, tz = getattr(cfg, f'{pre}tx'), getattr(cfg, f'{pre}ty'), getattr(cfg, f'{pre}tz')
    tx, ty, tz = _ask_vec3(f"Translation camera {cam_id}  tx ty tz", tx, ty, tz, unit='m')
    setattr(cfg, f'{pre}tx', tx)
    setattr(cfg, f'{pre}ty', ty)
    setattr(cfg, f'{pre}tz', tz)


def _edit_focal(cfg):
    cfg.fx = _ask_float("Focale fx (px)", cfg.fx, vmin=1)
    cfg.fy = _ask_float("Focale fy (px)", cfg.fy, vmin=1)


def _edit_principal(cfg):
    cfg.cx = _ask_float("Point principal cx (px)", cfg.cx, vmin=0)
    cfg.cy = _ask_float("Point principal cy (px)", cfg.cy, vmin=0)


def _edit_resolution(cfg):
    cfg.img_w = _ask_int("Largeur image W (px)", cfg.img_w, vmin=64)
    cfg.img_h = _ask_int("Hauteur image H (px)", cfg.img_h, vmin=64)
    cfg.cx    = _ask_float("Point principal cx (px)", cfg.img_w / 2, vmin=0)
    cfg.cy    = _ask_float("Point principal cy (px)", cfg.img_h / 2, vmin=0)


def _pick_mode(cfg):
    _cls()
    print("  -- MODE D'ANALYSE --\n")
    print("  [1] Scene unique     une scene, visualisation 3D complete")
    print("  [2] Balayage bruit   courbes erreur vs sigma (meme seed)")
    print("  [3] Monte-Carlo      distribution sur N scenes independantes\n")
    c = input("  Choix : ").strip()
    if c == '1':
        cfg.mode = 'single'
    elif c == '2':
        cfg.mode = 'noise_sweep'
        print()
        s = input(f"  Niveaux sigma (espaces) [{' '.join(str(x) for x in cfg.noise_levels)}] : ").strip()
        if s:
            try:
                cfg.noise_levels = sorted([float(x) for x in s.split()])
            except ValueError:
                print("  Valeurs invalides conservees.")
    elif c == '3':
        cfg.mode = 'montecarlo'
        cfg.mc_n = _ask_int("Nombre de scenes N", cfg.mc_n, vmin=5, vmax=10000)
    print()
    cfg.export_png = _ask_yesno("Sauvegarder les figures en PNG", cfg.export_png)


# ══════════════════════════════════════════════════════════════════════════════
#  Affichage du menu principal
# ══════════════════════════════════════════════════════════════════════════════

_SCENE_LABELS = {
    'planar':    'Planaire       Z = {z_min:.2f} m',
    'nonplanar': 'Non-planaire   Z in [3, 7] m',
    'custom':    'Personnalisee  Z in [{z_min:.2f}, {z_max:.2f}] m',
}
_MODE_LABELS = {
    'single':      'Scene unique',
    'noise_sweep': 'Balayage bruit',
    'montecarlo':  'Monte-Carlo',
}


def _fmt_scene(cfg):
    return _SCENE_LABELS[cfg.scene_type].format(z_min=cfg.z_min, z_max=cfg.z_max)


def _yn(b):
    return 'OUI' if b else 'NON'


def _print_menu(cfg):
    _cls()
    W = 64
    print('=' * W)
    print('   CAMERA POSE ESTIMATION  —  Outil de Recherche Interactif')
    print('=' * W)

    def row(num, label, value):
        tag = f'[{num:>2}]' if num != '' else '     '
        print(f'  {tag}  {label:<34}  {value}')

    print(f'\n  -- SCENE {"-" * (W - 11)}')
    row(1,  'Type de scene',           _fmt_scene(cfg))
    row(2,  'Etendue XY (m)',          f'+/-{cfg.x_range:.2f}  x  +/-{cfg.y_range:.2f}')
    row(3,  'Nombre de points',        cfg.n_points)
    row(4,  'Bruit pixel sigma (px)',  f'{cfg.noise_sigma:.3f}')
    row(5,  'Outliers',                f'{cfg.outlier_ratio:.0%}')
    row(6,  'Seed aleatoire',          cfg.seed)

    print(f'\n  -- CAMERA 1  (reference) {"-" * (W - 27)}')
    row(7,  'Rotation   rx ry rz (deg)',
        f'{cfg.cam1_rx:.1f}  {cfg.cam1_ry:.1f}  {cfg.cam1_rz:.1f}')
    row(8,  'Translation tx ty tz (m)',
        f'{cfg.cam1_tx:.3f}  {cfg.cam1_ty:.3f}  {cfg.cam1_tz:.3f}')

    print(f'\n  -- CAMERA 2 {"-" * (W - 14)}')
    row(9,  'Rotation   rx ry rz (deg)',
        f'{cfg.cam2_rx:.1f}  {cfg.cam2_ry:.1f}  {cfg.cam2_rz:.1f}')
    row(10, 'Translation tx ty tz (m)',
        f'{cfg.cam2_tx:.3f}  {cfg.cam2_ty:.3f}  {cfg.cam2_tz:.3f}')

    print(f'\n  -- INTRINSEQUES {"-" * (W - 18)}')
    row(11, 'Focale fx  fy (px)',         f'{cfg.fx:.0f}  {cfg.fy:.0f}')
    row(12, 'Point principal cx  cy (px)', f'{cfg.cx:.0f}  {cfg.cy:.0f}')
    row(13, 'Resolution W x H (px)',       f'{cfg.img_w}  x  {cfg.img_h}')

    print(f'\n  -- METHODES {"-" * (W - 14)}')
    row(14, 'Homographie H',              _yn(cfg.use_H) + '  (toggle)')
    row(15, 'Matrice fondamentale F',     _yn(cfg.use_F) + '  (toggle)')

    print(f'\n  -- MODE D\'ANALYSE {"-" * (W - 20)}')
    row(16, 'Mode',                       _MODE_LABELS[cfg.mode])
    if cfg.mode == 'noise_sweep':
        row('', 'Niveaux sigma',          ' '.join(f'{x:.1f}' for x in cfg.noise_levels))
    if cfg.mode == 'montecarlo':
        row('', 'Scenes N',               cfg.mc_n)
    row('', 'Export PNG',                 _yn(cfg.export_png))

    print()
    print('  ' + '-' * (W - 2))
    print('  [R] Lancer     [D] Valeurs par defaut     [Q] Quitter')
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Boucle principale
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = Config()

    dispatch = {
        '1':  lambda: _pick_scene_type(cfg),
        '2':  lambda: _edit_xy_range(cfg),
        '3':  lambda: setattr(cfg, 'n_points',
                              _ask_int("Nombre de points", cfg.n_points, vmin=8)),
        '4':  lambda: setattr(cfg, 'noise_sigma',
                              _ask_float("Bruit sigma (px)", cfg.noise_sigma, vmin=0)),
        '5':  lambda: setattr(cfg, 'outlier_ratio',
                              _ask_float("Ratio outliers  0.0=aucun  1.0=100%",
                                        cfg.outlier_ratio, vmin=0.0, vmax=1.0)),
        '6':  lambda: setattr(cfg, 'seed',
                              _ask_int("Seed", cfg.seed)),
        '7':  lambda: _edit_rotation(cfg, 1),
        '8':  lambda: _edit_translation(cfg, 1),
        '9':  lambda: _edit_rotation(cfg, 2),
        '10': lambda: _edit_translation(cfg, 2),
        '11': lambda: _edit_focal(cfg),
        '12': lambda: _edit_principal(cfg),
        '13': lambda: _edit_resolution(cfg),
        '14': lambda: setattr(cfg, 'use_H', not cfg.use_H),
        '15': lambda: setattr(cfg, 'use_F', not cfg.use_F),
        '16': lambda: _pick_mode(cfg),
    }

    while True:
        _print_menu(cfg)
        choice = input('  Votre choix : ').strip().lower()

        if choice in ('q', 'quit', 'exit'):
            print("\n  Au revoir !\n")
            break

        elif choice in ('d', 'default', 'defaut', 'defaults'):
            cfg = Config()
            print("  Configuration reinitalisee aux valeurs par defaut.")

        elif choice in ('r', 'run', 'lancer'):
            if not cfg.use_H and not cfg.use_F:
                print("  Activez au moins une methode ([14] ou [15]).")
                input("  [Entree pour continuer]")
                continue
            print()
            if   cfg.mode == 'single':
                analyse_single(cfg)
            elif cfg.mode == 'noise_sweep':
                analyse_noise_sweep(cfg)
            elif cfg.mode == 'montecarlo':
                analyse_montecarlo(cfg)
            input("\n  [Entree pour revenir au menu]")

        elif choice in dispatch:
            dispatch[choice]()


if __name__ == '__main__':
    main()
