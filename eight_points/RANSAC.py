import numpy as np
import math

class RANSAC_F:
    def __init__(self, s, epsilon, eight_points, px1, px2):
        self.s = s #Number of sample points
        self.epsilon = epsilon #Proportion of outliers
        self.eigh_points = eight_points
        self.px1 = px1
        self.px2 = px2
        self.best_F = None
        self.best_score = -1
        self.N = None
        self.best_mask = None

    def sample_size(self):
        p = 0.99
        if self.epsilon == 0.0:
            self.N = 1
        else:
            N = math.log(1 - p)/math.log(1 - (1 - self.epsilon)**self.s)
            self.N = int(math.ceil(N))

            
    
    def random_samples(self):
        n = self.px1.shape[1]
        idx = np.random.choice(n, self.s, replace=False)
        sample_px1 = self.px1[:, idx]
        sample_px2 = self.px2[:, idx]
        if sample_px1.shape[0] != 3:
            sample_px1 = np.vstack((sample_px1, np.ones((1, self.s))))
            sample_px2 = np.vstack((sample_px2, np.ones((1, self.s))))
        return sample_px1, sample_px2

    def execute_RANSAC(self):
        self.sample_size()
        for i in range(self.N):
            sample_px1, sample_px2 = self.random_samples()
            eight_point_input = (sample_px1, sample_px2)
            F_candidate = self.eigh_points(*eight_point_input)
            if F_candidate is None:
                continue
            current_score, current_mask = score_F(F_candidate, self.px1, self.px2, threshold=3.84)
            if current_score >self.best_score:
                self.best_score = current_score
                self.best_F = F_candidate
                self.best_mask = current_mask
        if self.best_mask is None or np.sum(self.best_mask) < 8:
            print("Warning: RANSAC failed to find enough inliers.")
            return self.best_F, self.best_mask 
            
        inlier_px1 = self.px1[:, self.best_mask]
        inlier_px2 = self.px2[:, self.best_mask]
        inlier_px1 = np.vstack((inlier_px1, np.ones((1, inlier_px1.shape[1]))))
        inlier_px2 = np.vstack((inlier_px2, np.ones((1, inlier_px1.shape[1]))))
        
        final_input = (inlier_px1, inlier_px2)
        final_F = self.eigh_points(*final_input)
        
        if final_F is not None:
            self.best_F = final_F
            
        return self.best_F, self.best_mask

def score_F(F, px1, px2, threshold=3.84):
    """
    Symmetric Sampson distance score for fundamental matrix F (ORB-SLAM style).

    Sampson distance is the first-order approximation of the reprojection error
    under the epipolar constraint x2ᵀFx1 = 0 :

      d²_sampson = (x2ᵀFx1)² / (‖Fx1‖²_top2 + ‖Fᵀx2‖²_top2)

      S_F = Σ max(0, T - d²_sampson)

    threshold : chi² at 95% for 1 DOF = 3.84  (epipolar constraint is 1D)

    Returns: float  (higher = better fit)
    """
    N = px1.shape[1]
    h1 = np.vstack([px1, np.ones((1, N))])
    h2 = np.vstack([px2, np.ones((1, N))])

    Fx1  = F @ h1                                              # (3, N)
    Ftx2 = F.T @ h2                                           # (3, N)

    num      = np.sum(h2 * Fx1, axis=0) ** 2                  # (x2ᵀFx1)²
    denom    = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
    d_samp   = num / (denom + 1e-12)
    score = float(np.sum(np.maximum(0, threshold - d_samp)))
    inlier_mask = d_samp < threshold
    return score, inlier_mask

        