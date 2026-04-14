F = eight_point(points1.T, points2.T) # Calculate F first then deduce E
tf, R_1f, R_2f = get_R_t_from_epipolar(F, K = K) 
E = eight_point(points1.T, points2.T, K1 = K, K2 = K) # Calculate E directly
te, R_1e, R_2e = get_R_t_from_epipolar(E, K = None)