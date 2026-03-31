import numpy as np

def project_points(points_3d, K, R, t):
    """
    points_3d : (3, N)
    K         : (3, 3)
    R         : (3, 3)
    t         : (3,)
    return  : pixels (2, N) and depth (N,)
    """

    X_cam = R @ points_3d + t.reshape(3, 1)  # (3, N)
    Z = X_cam[2, :]   # depths (N,)

    # normalize the coord with the depth info
    x_norm = X_cam[0, :] / Z
    y_norm = X_cam[1, :] / Z

    #apply the intrinsic matrix K
    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]

    return np.stack([u, v], axis=0), Z  # (2, N), (N,)


def filter_visible(pixels, depths, W=1920, H=1080):
    """
    pixels : (2, N)
    retourne masque booléen des pixels visibles des dimensions de la caméra (N,)
    """
    return ((depths > 0) & (pixels[0, :] >= 0) & (pixels[0, :] < W) & (pixels[1, :] >= 0) & (pixels[1, :] < H))
