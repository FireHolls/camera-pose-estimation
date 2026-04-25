"""
3-D visualisation of the camera pose estimation pipeline.

Layout per scenario:
  Left  (large) : 3-D world view — point cloud + camera frustums (GT and estimated)
  Right col 0   : image plane of camera 1 (GT projections)
  Right col 1   : image plane of camera 2 — H reprojection vs GT
  Right col 2   : image plane of camera 2 — F reprojection vs GT
  Right col 3   : score bar chart (S_H vs S_F)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

# ── Default image resolution (overridable via img_shape kwarg) ─────────────────
W_IMG, H_IMG  = 1920, 1080
FRUSTUM_DEPTH = 2.5
AXIS_LEN      = 0.30

CAM1_C = '#1976D2'
CAM2_C = '#388E3C'
H_C    = '#F57C00'
F_C    = '#D32F2F'


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _cam_center(R, t):
    return -R.T @ t


def _frustum_corners(R, t, K, depth=FRUSTUM_DEPTH, w=W_IMG, h=H_IMG):
    C = _cam_center(R, t)
    corners_px = [(0, 0), (w, 0), (w, h), (0, h)]
    out = []
    for u, v in corners_px:
        p_cam = np.array([(u - K[0, 2]) / K[0, 0],
                          (v - K[1, 2]) / K[1, 1],
                          1.0]) * depth
        out.append(R.T @ p_cam + C)
    return np.array(out)


def _draw_frustum(ax, R, t, K, color, label, lw=1.8, ls='-', w=W_IMG, h=H_IMG):
    C       = _cam_center(R, t)
    corners = _frustum_corners(R, t, K, w=w, h=h)

    for corner in corners:
        ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]],
                color=color, lw=lw, ls=ls, alpha=0.9)
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color=color, lw=lw, ls=ls, alpha=0.9)

    ax.scatter(*C, color=color, s=60, zorder=6, depthshade=False)
    fwd = R.T @ np.array([0., 0., 1.]) * AXIS_LEN * 2.5
    ax.quiver(*C, *fwd, color=color, lw=2.2, arrow_length_ratio=0.3, label=label)


def _draw_cam_axes(ax, R, t):
    C = _cam_center(R, t)
    for i, col in enumerate(['#E53935', '#43A047', '#1E88E5']):
        d = R.T[:, i] * AXIS_LEN
        ax.quiver(*C, *d, color=col, lw=1.2, arrow_length_ratio=0.4, alpha=0.8)


def _project(pts3d, R, t, K):
    X_cam = R @ pts3d + t[:, None]
    Z     = X_cam[2]
    valid = Z > 0
    u = K[0, 0] * X_cam[0, valid] / Z[valid] + K[0, 2]
    v = K[1, 1] * X_cam[1, valid] / Z[valid] + K[1, 2]
    return u, v


# ── Equal-scale 3-D axes ───────────────────────────────────────────────────────

def _set_equal_3d(ax, *point_sets):
    all_pts = np.hstack([p[:, None] if p.ndim == 1 else p for p in point_sets])
    mn, mx  = all_pts.min(1), all_pts.max(1)
    centre  = (mn + mx) / 2
    half    = (mx - mn).max() / 2 * 1.2
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass


# ── Sub-plot drawing functions ─────────────────────────────────────────────────

def _draw_3d(ax, scene, R_H, t_H, R_F, t_F, winner, planar, w=W_IMG, h=H_IMG):
    pts3d  = scene['pts3d']
    K      = scene['K']
    R2, t2 = scene['R_rel'], scene['t_rel']
    R1, t1 = np.eye(3), np.zeros(3)

    ax.scatter(pts3d[0], pts3d[1], pts3d[2],
               c=pts3d[2], cmap='plasma', s=12, alpha=0.45, label='Points 3D')

    _draw_frustum(ax, R1, t1, K, CAM1_C, 'Camera 1 (GT)', lw=2.0, w=w, h=h)
    _draw_cam_axes(ax, R1, t1)

    _draw_frustum(ax, R2, t2, K, CAM2_C, 'Camera 2 (GT)', lw=2.0, w=w, h=h)
    _draw_cam_axes(ax, R2, t2)

    if R_H is not None and t_H is not None:
        lw  = 2.4 if winner == 'H' else 1.2
        ls  = '-'  if winner == 'H' else '--'
        lbl = 'Camera 2 H' + ('  [VAINQUEUR]' if winner == 'H' else '')
        _draw_frustum(ax, R_H, t_H, K, H_C, lbl, lw=lw, ls=ls, w=w, h=h)

    if R_F is not None and t_F is not None:
        lw  = 2.4 if winner == 'F' else 1.2
        ls  = '-'  if winner == 'F' else '--'
        lbl = 'Camera 2 F' + ('  [VAINQUEUR]' if winner == 'F' else '')
        _draw_frustum(ax, R_F, t_F, K, F_C, lbl, lw=lw, ls=ls, w=w, h=h)

    ax.set_xlabel('X (m)', fontsize=8, labelpad=4)
    ax.set_ylabel('Y (m)', fontsize=8, labelpad=4)
    ax.set_zlabel('Z (m)', fontsize=8, labelpad=4)
    scene_lbl = 'Planaire' if planar else 'Non-planaire'
    ax.set_title(f'Vue 3D — {scene_lbl}', fontsize=10, pad=10)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.7)
    ax.view_init(elev=22, azim=-65)

    C1 = _cam_center(R1, t1)
    C2 = _cam_center(R2, t2)
    extra = [C1, C2]
    if R_H is not None and t_H is not None:
        extra.append(_cam_center(R_H, t_H))
    if R_F is not None and t_F is not None:
        extra.append(_cam_center(R_F, t_F))
    _set_equal_3d(ax, pts3d, *[e[:, None] for e in extra])


def _format_image_ax(ax, title, w=W_IMG, h=H_IMG):
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('u (px)', fontsize=8)
    ax.set_ylabel('v (px)', fontsize=8)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.7)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)


def _draw_image_cam1(ax, px1, w=W_IMG, h=H_IMG):
    ax.scatter(px1[0], px1[1], s=8, c=CAM1_C, alpha=0.6, label='GT')
    _format_image_ax(ax, 'Camera 1 — plan image', w, h)


def _draw_image_cam2_H(ax, px2, pts3d, R_H, t_H, K, winner, w=W_IMG, h=H_IMG):
    ax.scatter(px2[0], px2[1], s=8, c=CAM2_C, alpha=0.55, label='GT', zorder=2)
    if R_H is not None and t_H is not None:
        u, v = _project(pts3d, R_H, t_H, K)
        lbl  = 'Reproj. H' + (' [vainqueur]' if winner == 'H' else '')
        ax.scatter(u, v, s=9, c=H_C, marker='x', alpha=0.85, label=lbl, zorder=3)
    suffix = '  [VAINQUEUR]' if winner == 'H' else ''
    _format_image_ax(ax, f'Camera 2 — Reprojection H{suffix}', w, h)


def _draw_image_cam2_F(ax, px2, pts3d, R_F, t_F, K, winner, w=W_IMG, h=H_IMG):
    ax.scatter(px2[0], px2[1], s=8, c=CAM2_C, alpha=0.55, label='GT', zorder=2)
    if R_F is not None and t_F is not None:
        u, v = _project(pts3d, R_F, t_F, K)
        lbl  = 'Reproj. F' + (' [vainqueur]' if winner == 'F' else '')
        ax.scatter(u, v, s=9, c=F_C, marker='+', alpha=0.85, label=lbl, zorder=3)
    suffix = '  [VAINQUEUR]' if winner == 'F' else ''
    _format_image_ax(ax, f'Camera 2 — Reprojection F{suffix}', w, h)


def _draw_score_bar(ax, S_H, S_F, winner, ratio, thresh):
    total = S_H + S_F
    bars  = ax.bar(['Score H', 'Score F'], [S_H, S_F],
                   color=[H_C, F_C], alpha=0.82, width=0.5)
    win_idx = 0 if winner == 'H' else 1
    bars[win_idx].set_edgecolor('black')
    bars[win_idx].set_linewidth(2.5)

    ax.axhline(thresh * total, color='#616161', ls='--', lw=1.2,
               label=f'Seuil H  (ratio = {thresh})')
    for bar, val in zip(bars, [S_H, S_F]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.012,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    ax.set_title(f'Scores ORB-SLAM   ratio = {ratio:.3f}  ->  [{winner}]', fontsize=9)
    ax.set_ylabel('Score (plus haut = meilleur)', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, axis='y', alpha=0.25)
    ax.tick_params(labelsize=8)


# ── Public entry point ─────────────────────────────────────────────────────────

def plot_scene_3d(scene, R_H, t_H, R_F, t_F, winner, S_H, S_F, ratio, thresh,
                  planar, img_shape=(W_IMG, H_IMG)):
    """
    Full visual report for one scenario.

    Parameters
    ----------
    scene      : dict from make_scene()  (pts3d, px1, px2, K, R_rel, t_rel)
    R_H, t_H   : estimated camera-2 pose from homography  (None if unavailable)
    R_F, t_F   : estimated camera-2 pose from 8-point      (None if unavailable)
    winner     : 'H' or 'F'
    S_H, S_F   : ORB-SLAM scores
    ratio      : S_H / (S_H + S_F)
    thresh     : decision threshold (0.45)
    planar     : bool
    img_shape  : (W, H) image resolution used to set image-plane axes
    """
    w, h  = img_shape
    pts3d = scene['pts3d']
    px1   = scene['px1']
    px2   = scene['px2']
    K     = scene['K']

    fig = plt.figure(figsize=(17, 11))
    gs  = gridspec.GridSpec(
        4, 2, figure=fig,
        width_ratios=[2.2, 1.0],
        height_ratios=[1, 1, 1, 1],
        hspace=0.55, wspace=0.30,
    )

    ax3d   = fig.add_subplot(gs[:, 0], projection='3d')
    ax_c1  = fig.add_subplot(gs[0, 1])
    ax_c2h = fig.add_subplot(gs[1, 1])
    ax_c2f = fig.add_subplot(gs[2, 1])
    ax_bar = fig.add_subplot(gs[3, 1])

    scene_lbl = 'Planaire (Z=5)' if planar else 'Non-planaire (Z in [3,7])'
    fig.suptitle(
        f'Camera Pose Estimation  —  Scene {scene_lbl}   |   Vainqueur : [{winner}]',
        fontsize=13, fontweight='bold',
    )

    _draw_3d(ax3d, scene, R_H, t_H, R_F, t_F, winner, planar, w=w, h=h)
    _draw_image_cam1(ax_c1, px1, w=w, h=h)
    _draw_image_cam2_H(ax_c2h, px2, pts3d, R_H, t_H, K, winner, w=w, h=h)
    _draw_image_cam2_F(ax_c2f, px2, pts3d, R_F, t_F, K, winner, w=w, h=h)
    _draw_score_bar(ax_bar, S_H, S_F, winner, ratio, thresh)

    plt.tight_layout()
    plt.show()
