import numpy as np


def _normalize_2d(points_2d):
    """
    Normalisation pour les coordonnées 2D 

    Sans normalisation, la matrice A du DLT est très mal conditionnée
    et eviter d'amplifier les erreurs numériques
    lors du calcul SVD.


    La transformation est T = [[s, 0, -s·cx],
                                [0, s, -s·cy],
                                [0, 0,    1 ]]

    entrée
    points_2d : (2, N) ndarray — coordonnées pixel

    Retourne
    pts_norm : (2, N) ndarray — coordonnées normalisées
    T        : (3, 3) ndarray — matrice de normalisation telle que
                                x̃_norm = T @ x̃_homog
    """
    N = points_2d.shape[1]

    # Étape 1 : translation au centroïde
    cx = points_2d[0].mean()
    cy = points_2d[1].mean()
    shifted = points_2d - np.array([[cx], [cy]])  # (2, N)

    # Étape 2 : facteur d'échelle → distance moyenne = sqrt(2)
    mean_dist = np.mean(np.sqrt(shifted[0] ** 2 + shifted[1] ** 2))
    s = np.sqrt(2) / mean_dist

    T = np.array([
        [s, 0, -s * cx],
        [0, s, -s * cy],
        [0, 0,       1]
    ], dtype=np.float64)

    # Appliquer T via les coordonnées homogènes
    pts_h = np.vstack([points_2d, np.ones((1, N))])  # (3, N)
    pts_norm = T @ pts_h                              # (3, N)

    return pts_norm[:2], T


def _normalize_3d(points_3d):
    """
    Normalisation  pour les coordonnées 3D.

    La transformation est T = [[s, 0, 0, -s·cx],
                                [0, s, 0, -s·cy],
                                [0, 0, s, -s·cz],
                                [0, 0, 0,    1 ]]

    entrée
    points_3d : (3, N) ndarray — coordonnées monde 3D

    Retourne
    pts_norm : (3, N) ndarray — coordonnées normalisées
    T        : (4, 4) ndarray — matrice de normalisation telle que
                                X̃_norm = T @ X̃_homog
    """
    N = points_3d.shape[1]

    cx = points_3d[0].mean()
    cy = points_3d[1].mean()
    cz = points_3d[2].mean()
    shifted = points_3d - np.array([[cx], [cy], [cz]])  # (3, N)

    mean_dist = np.mean(np.sqrt(shifted[0] ** 2 + shifted[1] ** 2 + shifted[2] ** 2))
    s = np.sqrt(3) / mean_dist

    T = np.array([
        [s, 0, 0, -s * cx],
        [0, s, 0, -s * cy],
        [0, 0, s, -s * cz],
        [0, 0, 0,       1]
    ], dtype=np.float64)

    pts_h = np.vstack([points_3d, np.ones((1, N))])  # (4, N)
    pts_norm = T @ pts_h                              # (4, N)

    return pts_norm[:3], T


def _build_A(points_2d, points_3d):
    """
    Construit la matrice A de taille (2N × 12) pour le système homogène A p = 0.

    entrée
    points_2d : (2, N) ndarray
    points_3d : (3, N) ndarray

    Retourne
    A : (2N, 12) ndarray
    """
    N = points_3d.shape[1]
    A = np.zeros((2 * N, 12), dtype=np.float64)

    for i in range(N):
        X, Y, Z = points_3d[:, i]
        u, v = points_2d[:, i]

        # Ligne 2i   : équation en u → [−X̃ᵀ, 0ᵀ, u·X̃ᵀ]
        A[2 * i,     :] = [-X, -Y, -Z, -1,  0,  0,  0,  0,  u*X,  u*Y,  u*Z,  u]
        # Ligne 2i+1 : équation en v → [0ᵀ, −X̃ᵀ, v·X̃ᵀ]
        A[2 * i + 1, :] = [ 0,  0,  0,  0, -X, -Y, -Z, -1,  v*X,  v*Y,  v*Z,  v]

    return A


def dlt(points_2d, points_3d, normalize=True):
    """
    Estime la matrice de projection 3×4  P = K[R|t]  a partir de N ≥ 6 correspondances 2D↔3D.


    points_2d : (2, N) ndarray — coordonnées pixel observées
    points_3d : (3, N) ndarray — coordonnées 3D monde correspondantes
    normalize : bool           — appliquer la normalisation 

    Retourne
    P : (3, 4) ndarray — matrice de projection estimée, normalisée ‖P‖_F = 1
    """
    N = points_3d.shape[1]
    assert N >= 6, (
        f"Le DLT nécessite au moins 6 correspondances, reçu {N}. "
        "P a 11 DDL ; chaque correspondance donne 2 équations → ⌈11/2⌉ = 6 minimum."
    )

    if normalize:
        pts2d_n, T2d = _normalize_2d(points_2d)
        pts3d_n, T3d = _normalize_3d(points_3d)
    else:
        pts2d_n, pts3d_n = points_2d, points_3d

    # Construction du système A p = 0  (2N × 12)
    A = _build_A(pts2d_n, pts3d_n)

    # Résolution par SVD : A = U Σ Vᵀ
    # La solution p est la dernière ligne de Vᵀ (= dernière colonne de V),
    _, _, Vt = np.linalg.svd(A)
    p = Vt[-1]          # (12,) — vecteur solution
    P_norm = p.reshape(3, 4)

    if normalize:
        # Dénormalisation : P_true = T2d⁻¹ @ P_norm @ T3d
        P = np.linalg.inv(T2d) @ P_norm @ T3d
    else:
        P = P_norm

    if np.linalg.det(P[:, :3]) < 0:
        P = -P

    P /= np.linalg.norm(P)

    return P


def extract_Rt_from_P(P, K):
    """
    Extrait R (rotation) et t (translation) depuis la matrice de projection P = K[R|t],

    Principe :
    P = K [R | t]  →  K⁻¹ P = [R | t]  


    Paramètres

    P : (3, 4) ndarray — matrice de projection estimée par DLT
    K : (3, 3) ndarray — matrice intrinsèque connue

    Retourne

    R : (3, 3) ndarray — matrice de rotation (det = +1, colonnes orthonormées)
    t : (3,)   ndarray — vecteur de translation
    """
    # Étape 1 : annuler K pour obtenir [R | t] à l'échelle λ près
    Rt = np.linalg.inv(K) @ P   # (3, 4)  ≈ λ [R | t]

    R_approx = Rt[:, :3]        # (3, 3)  ≈ λ R
    t_approx = Rt[:, 3]         # (3,)    ≈ λ t

    # np.linalg.svd retourne U, s, Vh  où  Vh = Vᵀ 
    U, s, Vh = np.linalg.svd(R_approx)

    # Étape 3 : projection R = U Vᵀ
    R = U @ Vh

    # Étape 4 : correction  det(R) doit être +1
    if np.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = U @ Vh

    # Étape 5 : récupérer l'échelle λ depuis 
    scale = np.mean(s)
    t = t_approx / scale

    return R, t
