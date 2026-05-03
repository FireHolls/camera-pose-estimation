import numpy as np
import math

class RANSAC:
    def __init__(self, s, score_fct, model_fct, px1, px2, epsilon = None):
        self.s = s #Number of sample points
        self.score_fct = score_fct #The score function
        self.model_fct = model_fct #The model function
        self.px1 = px1 #2D points in image 1 (2xN)
        self.px2 = px2 #2D points in image 2 (2xN)
        self.epsilon = epsilon #Proportion of outliers
        self.best_model = None
        self.best_score = -1
        self.N = None #Number of selections to be determined
        self.best_mask = None
        self.nb_inlier = -1
        self.rng = np.random.default_rng()
        self.log_prob = math.log(1.0 - 0.99)
        self.n_pts = px1.shape[1]

    def selections(self):
        """
        Function to determine the number of selection N, where N = log(1 - p)/log(1 - (1 - epsilon)^s),
        where s is the sample size, epsilon is the probability that any data point is an outlier and p 
        is the probability that at least one of the random samples of s points is free from outliers.
        """
        if self.epsilon == 0.0:
            self.N = 1
        else:
            N = self.log_prob/math.log(1 - (1 - self.epsilon)**self.s)
            self.N = int(math.ceil(N))
    
    def random_samples(self):
        """
        Function to select a random sample of points to run the model.
        """
        n = self.px1.shape[1]
        idx = np.random.choice(n, self.s, replace=False)
        sample_px1 = self.px1[:, idx]
        sample_px2 = self.px2[:, idx]
        return sample_px1, sample_px2

    def execute_RANSAC(self):
        """
        Function to execute the RANSAC and find the largest valid set and the model which matches this set
        """
        if self.epsilon is not None:
            self.selections()
            max_iterations = self.N
        else:   
            max_iterations = 100000
        iteration = 0
        while iteration < max_iterations:
            #1 Generate model
            idx = self.rng.choice(self.n_pts, self.s, replace=False)
            sample_px1 = self.px1[:, idx]
            sample_px2 = self.px2[:, idx]
            candidate = self.model_fct(sample_px1, sample_px2)
            if candidate is None:
                continue
            #2 Evaluate
            current_score, current_mask = self.score_fct(candidate, self.px1, self.px2)
            current_inlier_count = np.sum(current_mask)
            #3 Save if it's the best model
            if current_score >self.best_score:
                self.best_score = current_score
                self.best_model = candidate
                self.best_mask = current_mask
                if self.epsilon is None or current_inlier_count > self.nb_inlier:
                    self.nb_inlier = current_inlier_count
                    self.epsilon = 1.0 - (self.nb_inlier / self.px1.shape[1])
                    self.selections()
                    max_iterations = min(max_iterations, self.N)
            iteration += 1
        if self.best_mask is None or np.sum(self.best_mask) < 8:
            print("Warning: RANSAC failed to find enough inliers.")
            return self.best_model, self.best_mask 
            
        inlier_px1 = self.px1[:, self.best_mask]
        inlier_px2 = self.px2[:, self.best_mask]
        
        final_model = self.model_fct(inlier_px1, inlier_px2)
        
        if final_model is not None:
            self.best_model = final_model
            
        return self.best_model, self.best_mask
    
    

def score_F_RANSAC(F, px1, px2, threshold=3.84):
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

def score_H_RANSAC(H, px1, px2, threshold=5.99):
    """
    Symmetric transfer error score for homography H (ORB-SLAM style).

    For each correspondence, measures how well H maps px1→px2 AND H⁻¹ maps px2→px1.
    Points with error > threshold are considered outliers and contribute 0.

      S_H = Σ [ max(0, T - d²(H·x1, x2)) + max(0, T - d²(H⁻¹·x2, x1)) ]

    threshold : chi² at 95% for 2 DOF = 5.99  (transfer error is a 2D residual)

    Returns: float  (higher = better fit)
    """
    N = px1.shape[1]
    h1 = np.vstack([px1, np.ones((1, N))])   # (3, N)
    h2 = np.vstack([px2, np.ones((1, N))])   # (3, N)

    # Forward: H · px1 → px2
    p12 = H @ h1
    p12 = p12[:2] / p12[2]
    d12 = np.sum((p12 - px2) ** 2, axis=0)

    # Backward: H⁻¹ · px2 → px1
    p21 = np.linalg.inv(H) @ h2
    p21 = p21[:2] / p21[2]
    d21 = np.sum((p21 - px1) ** 2, axis=0)

    score_forward = np.maximum(0, threshold - d12)
    score_backward = np.maximum(0, threshold - d21)
    total_score = float(np.sum(score_forward + score_backward))

    inliers = (d12 < threshold) & (d21 < threshold)
    
    return total_score, inliers

        