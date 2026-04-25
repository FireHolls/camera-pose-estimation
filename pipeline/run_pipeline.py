"""
H vs F comparison pipeline — ORB/SLAM initialization style.

Runs on two scenarios:
  - Planar scene    (Z = 5)       → homography should win
  - Non-planar scene (Z ∈ [3,7]) → fundamental matrix should win

Usage (from project root):
    python pipeline/run_pipeline.py
"""

import sys, os, io, contextlib
sys.stdout.reconfigure(encoding='utf-8')
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, PIPELINE_DIR)

import numpy as np
import matplotlib.pyplot as plt

from scene import make_scene
from score import score_H, score_F

from simulation.homography import homography, decompose_H
from eight_points.eight_point_agl import eight_point
from eight_points.Retrieve_P import get_R_t_from_epipolar, P_estimation
from visualize3d import plot_scene_3d

PLANE_DIST      = 5.0   # depth of the scene plane in camera 1 frame (planar case)
H_RATIO_THRESH  = 0.45  # ORB-SLAM threshold: above → use H, below → use F



def _rot_err(R_est, R_true):
    """Angular error between two rotation matrices (degrees)."""
    trace = np.trace(R_est.T @ R_true)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def _trans_err(t_est, t_true):
    """Angular error between two translation directions (degrees).
    Uses abs(dot) to handle the global sign ambiguity of t from epipolar geometry."""
    u = t_est  / np.linalg.norm(t_est)
    v = t_true / np.linalg.norm(t_true)
    return np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), 0.0, 1.0)))


def _select_best_P(Ps, pts3d, px2, K):
    """
    Among the 4 candidate projection matrices from P_estimation,
    pick the one with the lowest reprojection RMSE on px2.
    Solutions where >50% of points are behind the camera are rejected first.

    Returns: R (3x3), t (3,)
    """
    K_inv    = np.linalg.inv(K)
    pts3d_h  = np.vstack([pts3d, np.ones((1, pts3d.shape[1]))])  # (4, M)
    best_rmse = np.inf
    best_R, best_t = None, None

    for P in Ps:
        proj   = P @ pts3d_h                  # (3, M)
        depths = proj[2]
        if (depths > 0).mean() < 0.5:         # mostly behind camera → invalid
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


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline for H  (homography via DLT)
# ──────────────────────────────────────────────────────────────────────────────

def run_H_pipeline(px1, px2, pts3d, K, planar):
    """
    1. Estimate H from px1 → px2 (DLT)
    2. Decompose H → R_rel, t_rel
    3. Compute score S_H

    plane_dist is passed only for planar scenes; otherwise t is returned as unit direction.
    """
    H = homography(px1, px2)
    S = score_H(H, px1, px2)

    X_ref      = pts3d[:, 0]
    plane_dist = PLANE_DIST if planar else None
    R, t       = decompose_H(H, K, plane_dist=plane_dist, X_ref=X_ref)

    return H, R, t, S


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline for F  (8-point algorithm)
# ──────────────────────────────────────────────────────────────────────────────

def run_F_pipeline(px1, px2, pts3d, K):
    """
    1. Estimate F from px1 → px2 (normalized 8-point + LM optimisation)
    2. Decompose F → E → 4 candidate (R, t)
    3. Select best candidate by reprojection RMSE
    4. Compute score S_F
    """
    N     = px1.shape[1]
    pts1h = np.vstack([px1, np.ones((1, N))]).T   # (N, 3)
    pts2h = np.vstack([px2, np.ones((1, N))]).T   # (N, 3)

    F = eight_point(pts1h, pts2h)                 # no K → returns F (pixel space)
    S = score_F(F, px1, px2)

    # Suppress the sigma print inside get_R_t_from_epipolar
    with contextlib.redirect_stdout(io.StringIO()):
        t_col, R1_f, R2_f = get_R_t_from_epipolar(F, K)   # K given → uses E = KᵀFK

    Ps    = P_estimation(t_col, R1_f, R2_f, K, s=1)        # (4, 3, 4)
    R, t  = _select_best_P(Ps, pts3d, px2, K)

    return F, R, t, S


# ──────────────────────────────────────────────────────────────────────────────
# Run one scenario and return results
# ──────────────────────────────────────────────────────────────────────────────

def run_scenario(planar):
    label = "PLANAIRE  (Z = 5)" if planar else "NON-PLANAIRE  (Z ∈ [3, 7])"
    scene = make_scene(planar=planar)
    pts3d, px1, px2, K = scene['pts3d'], scene['px1'], scene['px2'], scene['K']
    R_true, t_true = scene['R_rel'], scene['t_rel']
    M = px1.shape[1]

    print(f"\n{'='*60}")
    print(f"  Scène {label}  —  {M} points visibles")
    print(f"{'='*60}")

    # ── H pipeline
    _, R_H, t_H, S_H = run_H_pipeline(px1, px2, pts3d, K, planar)

    # ── F pipeline
    _, R_F, t_F, S_F = run_F_pipeline(px1, px2, pts3d, K)

    # ── Selection
    ratio   = S_H / (S_H + S_F)
    winner  = "H" if ratio > H_RATIO_THRESH else "F"

    # ── Errors
    err_R_H = _rot_err(R_H, R_true)   if R_H is not None else float('nan')
    err_t_H = _trans_err(t_H, t_true) if t_H is not None else float('nan')
    err_R_F = _rot_err(R_F, R_true)   if R_F is not None else float('nan')
    err_t_F = _trans_err(t_F, t_true) if t_F is not None else float('nan')

    print(f"\n  Scores")
    print(f"    S_H = {S_H:8.1f}    S_F = {S_F:8.1f}")
    print(f"    Ratio S_H/(S_H+S_F) = {ratio:.3f}  →  vainqueur : [{winner}]")
    print(f"\n  Erreurs vs vérité terrain (R_rel, t_rel)")
    print(f"    {'Méthode':6}  {'Err R (°)':>10}  {'Err t dir (°)':>14}")
    print(f"    {'H':6}  {err_R_H:>10.4f}  {err_t_H:>14.4f}")
    print(f"    {'F':6}  {err_R_F:>10.4f}  {err_t_F:>14.4f}")

    return {
        'label': label, 'planar': planar,
        'S_H': S_H, 'S_F': S_F, 'ratio': ratio, 'winner': winner,
        'err_R_H': err_R_H, 'err_t_H': err_t_H,
        'err_R_F': err_R_F, 'err_t_F': err_t_F,
        'scene': scene,
        'R_H': R_H, 't_H': t_H,
        'R_F': R_F, 't_F': t_F,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H vs F — Comparaison des scores (style ORB-SLAM)", fontsize=13, fontweight='bold')

    colors = {'H': 'steelblue', 'F': 'tomato'}

    for ax, res in zip(axes, results):
        S_H, S_F = res['S_H'], res['S_F']
        total    = S_H + S_F
        bars = ax.bar(['Score H', 'Score F'],
                      [S_H, S_F],
                      color=[colors['H'], colors['F']],
                      alpha=0.8, width=0.5)

        # Mark the winner
        winner_idx = 0 if res['winner'] == 'H' else 1
        bars[winner_idx].set_edgecolor('black')
        bars[winner_idx].set_linewidth(2.5)

        # Threshold line: S_H = 0.45 * (S_H + S_F)  ↔  S_H / total = 0.45
        ax.axhline(H_RATIO_THRESH * total, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Seuil H (ratio = {H_RATIO_THRESH})')

        ax.set_title(f"Scène {res['label']}\nVainqueur : [{res['winner']}]  "
                     f"(ratio = {res['ratio']:.3f})")
        ax.set_ylabel("Score (plus haut = meilleur fit)")
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        # Annotate bars
        for bar, val in zip(bars, [S_H, S_F]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = [
        run_scenario(planar=True),
        run_scenario(planar=False),
    ]

    print(f"\n{'='*60}")
    print("  RÉSUMÉ")
    print(f"{'='*60}")
    for r in results:
        print(f"  [{r['winner']:1}] gagne sur scène {r['label']}")

    for r in results:
        plot_scene_3d(
            r['scene'], r['R_H'], r['t_H'], r['R_F'], r['t_F'],
            r['winner'], r['S_H'], r['S_F'], r['ratio'], H_RATIO_THRESH,
            r['planar'],
        )
