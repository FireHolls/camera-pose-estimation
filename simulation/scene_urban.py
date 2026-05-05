import numpy as np


def _alloc(n, weights):
    total = sum(weights)
    counts = [max(1, int(round(n * w / total))) for w in weights]
    counts[0] += n - sum(counts)
    return counts


def _sample_box(rng, cx, cy_base, cz, width, depth, height, n):
    """Points uniformly distributed on the 6 faces of a box.
    Base center at (cx, cy_base, cz); box extends upward (Y decreasing).
    """
    x0, x1 = cx - width / 2,  cx + width / 2
    y_top   = cy_base - height
    y_bot   = cy_base
    z0, z1  = cz - depth / 2,  cz + depth / 2

    areas  = [
        width  * height,   # front  z=z1
        width  * height,   # back   z=z0
        width  * depth,    # top    y=y_top
        width  * depth,    # bottom y=y_bot
        height * depth,    # right  x=x1
        height * depth,    # left   x=x0
    ]
    counts = _alloc(n, areas)
    pts = []

    for z_val, cnt in [(z1, counts[0]), (z0, counts[1])]:
        if cnt > 0:
            x = rng.uniform(x0, x1, cnt)
            y = rng.uniform(y_top, y_bot, cnt)
            pts.append(np.vstack([x, y, np.full(cnt, z_val)]))

    for y_val, cnt in [(y_top, counts[2]), (y_bot, counts[3])]:
        if cnt > 0:
            x = rng.uniform(x0, x1, cnt)
            z = rng.uniform(z0, z1, cnt)
            pts.append(np.vstack([x, np.full(cnt, y_val), z]))

    for x_val, cnt in [(x1, counts[4]), (x0, counts[5])]:
        if cnt > 0:
            y = rng.uniform(y_top, y_bot, cnt)
            z = rng.uniform(z0, z1, cnt)
            pts.append(np.vstack([np.full(cnt, x_val), y, z]))

    return np.hstack(pts)


def _sample_tree(rng, cx, cy_base, cz, n):
    """Thin cylinder trunk + flattened hemisphere canopy (mushroom shape)."""
    trunk_r  = 0.15
    trunk_h  = 1.5
    canopy_r = 0.9   # horizontal radius
    canopy_h = 0.7   # vertical half-extent (flattened sphere)

    n_trunk  = max(1, n // 5)
    n_canopy = n - n_trunk

    # Trunk: points on cylinder surface
    angles = rng.uniform(0, 2 * np.pi, n_trunk)
    ys     = rng.uniform(cy_base - trunk_h, cy_base, n_trunk)
    pts_trunk = np.vstack([
        cx + trunk_r * np.cos(angles),
        ys,
        cz + trunk_r * np.sin(angles),
    ])

    # Canopy: upper hemisphere, flattened vertically
    canopy_cy = cy_base - trunk_h - canopy_h * 0.5
    phi   = rng.uniform(0, np.pi / 2, n_canopy)   # 0 = top, pi/2 = equator
    theta = rng.uniform(0, 2 * np.pi, n_canopy)
    pts_canopy = np.vstack([
        cx + canopy_r * np.sin(phi) * np.cos(theta),
        canopy_cy - canopy_h * np.cos(phi),
        cz + canopy_r * np.sin(phi) * np.sin(theta),
    ])

    return np.hstack([pts_trunk, pts_canopy])


def _sample_car(rng, cx, cy_base, cz, n):
    """Low body box + narrower roof box."""
    body_w, body_d, body_h = 2.0, 4.0, 1.2
    roof_w, roof_d, roof_h = 1.4, 2.5, 0.7

    n_body = max(1, int(n * 0.65))
    n_roof = n - n_body

    pts_body = _sample_box(rng, cx, cy_base,            cz, body_w, body_d, body_h, n_body)
    pts_roof = _sample_box(rng, cx, cy_base - body_h,   cz, roof_w, roof_d, roof_h, n_roof)

    return np.hstack([pts_body, pts_roof])


# ── Fixed scene layout ─────────────────────────────────────────────────────────
#  Camera at origin looking along +Z. Ground at Y = +1.5 (1.5 m below camera).

_BUILDINGS = [
    # (cx,  cz,  width, depth, height)
    ( 0.0, 10.0,   3.0,   3.0,   6.0),
]

_TREES = [
    # (cx, cz)
    (-3.0,  8.0),
]

_CARS = [
    # (cx, cz)
    ( 3.0,  7.0),
]

GROUND_Y = 1.5


BUILDING_COLOR = '#89b4fa'
TREE_COLOR     = '#a6e3a1'
CAR_COLOR      = '#fab387'


def generate_urban_scene(rng, n_points):
    """
    Builds a point cloud of a synthetic urban street scene.
    Returns (pts3d, colors) where pts3d is (3, N) and colors is (N,) of hex strings.
    """
    nb, nt, nc = len(_BUILDINGS), len(_TREES), len(_CARS)

    n_b = max(10, int(n_points * 0.55))
    n_t = max( 8, int(n_points * 0.25))
    n_c = max( 6, int(n_points * 0.20))

    all_pts    = []
    all_colors = []

    for (cx, cz, w, d, h) in _BUILDINGS:
        pts = _sample_box(rng, cx, GROUND_Y, cz, w, d, h, n_b)
        all_pts.append(pts)
        all_colors.extend([BUILDING_COLOR] * pts.shape[1])

    for (cx, cz) in _TREES:
        pts = _sample_tree(rng, cx, GROUND_Y, cz, n_t)
        all_pts.append(pts)
        all_colors.extend([TREE_COLOR] * pts.shape[1])

    for (cx, cz) in _CARS:
        pts = _sample_car(rng, cx, GROUND_Y, cz, n_c)
        all_pts.append(pts)
        all_colors.extend([CAR_COLOR] * pts.shape[1])

    pts3d  = np.hstack(all_pts)
    colors = np.array(all_colors)
    perm   = rng.permutation(pts3d.shape[1])
    return pts3d[:, perm], colors[perm]
