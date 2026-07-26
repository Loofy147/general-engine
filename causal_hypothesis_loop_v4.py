import numpy as np
from sklearn.linear_model import HuberRegressor

# ================================================================
# Ground Truth and Data Generation
# ================================================================
TRUE_K = 3.0
def ground_truth(x):
    return TRUE_K * x**2

def query_external_data(rng, n, x_range=(0.5, 5.0), noise_type="heteroscedastic"):
    """
    Improvement 3: Real-world noise.
    - 'heteroscedastic': noise scales with y
    - 'heavy_tailed': Cauchy-like noise components
    """
    x = rng.uniform(*x_range, size=n)
    y_true = ground_truth(x)
    
    if noise_type == "heteroscedastic":
        noise = rng.normal(0, 0.15 * y_true)
    elif noise_type == "heavy_tailed":
        # Mix of normal and some outliers
        noise = rng.normal(0, 0.10 * y_true) + rng.standard_cauchy(size=n) * 0.05 * y_true
    else:
        noise = rng.normal(0, 0.10 * np.abs(y_true))
        
    return x, y_true + noise

# ================================================================
# Multi-Modal Model Families (Improvement 1)
# ================================================================
class MultiModalModel:
    """
    A model that can represent y = sum(k_i * x^p_i).
    """
    def __init__(self, exponents):
        self.exponents = np.array(exponents)
        self.k = np.zeros(len(exponents))
        self.k_se = np.zeros(len(exponents))
        self.cov_matrix = None

    def __call__(self, x, k=None):
        if k is None: k = self.k
        # Handle both scalar/vector x and sample-based k (for simulator)
        # x shape: (N,) or (N, 1)
        # k shape: (M,) or (1, M)
        x = np.atleast_1d(x)
        
        if k.ndim == 1:
            # Standard prediction: sum(k_i * x^p_i)
            # Result shape: (N,)
            design_matrix = np.power(x[:, None], self.exponents[None, :])
            return design_matrix @ k
        else:
            # Simulator prediction with M samples of k
            # k shape: (len(exponents), M)
            # Result shape: (N, M)
            design_matrix = np.power(x[:, None], self.exponents[None, :])
            return design_matrix @ k

    def fit(self, x, y):
        """Robust regression for k given fixed exponents using Huber loss."""
        design_matrix = np.power(x[:, None], self.exponents[None, :])
        n, m = design_matrix.shape
        try:
            # Fit with Huber loss, no intercept
            huber = HuberRegressor(fit_intercept=False, max_iter=2000)
            huber.fit(design_matrix, y)
            self.k = huber.coef_
            
            # Robust Covariance Estimation from inliers
            inliers = ~huber.outliers_
            X_inliers = design_matrix[inliers]
            y_inliers = y[inliers]
            n_in, m_in = X_inliers.shape

            if n_in > m:
                # Calculate covariance matrix using inliers
                y_pred_in = X_inliers @ self.k
                mse_in = np.sum((y_inliers - y_pred_in)**2) / (n_in - m)
                self.cov_matrix = mse_in * np.linalg.inv(X_inliers.T @ X_inliers)
                self.k_se = np.sqrt(np.diag(self.cov_matrix))
            else:
                # Fallback to all data if there are too few inliers
                if n > m:
                    mse = np.sum((y - design_matrix @ self.k)**2) / (n - m)
                    self.cov_matrix = mse * np.linalg.inv(design_matrix.T @ design_matrix)
                    self.k_se = np.sqrt(np.diag(self.cov_matrix))
                else:
                    self.cov_matrix = None
                    self.k_se = np.zeros(m)

            # Robust residual scale estimation for simulator
            y_pred = design_matrix @ self.k
            rel_residuals = (y - y_pred) / (np.abs(y_pred) + 1e-9)
            mad = np.median(np.abs(rel_residuals - np.median(rel_residuals)))
            self.residual_scale = float(max(mad * 1.4826, 1e-4)) # fallback to prevent 0

        except Exception:
            # Fallback to simple least squares if HuberRegressor fails
            try:
                k_hat, residuals, rank, s = np.linalg.lstsq(design_matrix, y, rcond=None)
                self.k = k_hat
                if n > m:
                    mse = np.sum((y - design_matrix @ k_hat)**2) / (n - m)
                    self.cov_matrix = mse * np.linalg.inv(design_matrix.T @ design_matrix)
                    self.k_se = np.sqrt(np.diag(self.cov_matrix))
                else:
                    self.cov_matrix = None
                    self.k_se = np.zeros(m)

                # Simple relative residual scale fallback
                y_pred = design_matrix @ k_hat
                rel_residuals = (y - y_pred) / (np.abs(y_pred) + 1e-9)
                self.residual_scale = float(max(np.std(rel_residuals), 1e-4))
            except Exception:
                self.k = np.zeros(m)
                self.k_se = np.zeros(m)
                self.cov_matrix = None
                self.residual_scale = 0.10

# ================================================================
# Core Metrics
# ================================================================
def internal_simulator(rng, model, x_query, n_iter=2000):
    if model.cov_matrix is not None:
        # Sample k from the multivariate normal distribution
        k_samples = rng.multivariate_normal(model.k, model.cov_matrix, size=n_iter).T
    else:
        # Fallback to simple relative uncertainty if covariance is missing
        k_samples = rng.normal(model.k[:, None], np.abs(model.k[:, None]) * 0.1, size=(len(model.k), n_iter))
    
    y_pred = model(x_query, k=k_samples)
    # Add residual noise estimate robustly derived from the fit
    res_scale = getattr(model, 'residual_scale', 0.10)
    noise = rng.normal(0, res_scale * np.abs(y_pred) + 1e-9)
    return y_pred + noise

def coverage_verisimilitude(y_sim, ref_y):
    lo, hi = np.percentile(y_sim, [5, 95], axis=1)
    return float(np.mean((ref_y >= lo) & (ref_y <= hi)))

def r2_verisimilitude(model, ref_x, ref_y):
    y_pred = model(ref_x)
    ss_res = np.sum((ref_y - y_pred)**2)
    ss_tot = np.sum((ref_y - np.mean(ref_y))**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Adjusted R2 to penalize complexity (number of terms/exponents)
    n = len(ref_y)
    p = len(model.exponents)
    if n > p + 1:
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = 0.0
    return float(np.clip(adj_r2, 0.0, 1.0))

def max_error_ratio(model, ref_x, ref_y, eps=1e-12):
    y_pred = model(ref_x)
    return float(np.max(np.abs(ref_y - y_pred) / (np.abs(ref_y) + eps)))

def hybrid_verisimilitude(r2, cov, mer, lam=0.25):
    harmonic = (2 * max(0, r2) * cov / (max(0, r2) + cov)) if (max(0, r2) + cov) > 0 else 0.0
    return float(np.clip(harmonic - lam * mer, 0.0, 1.0))

# ================================================================
# Layer 4: Active Data Acquisition (Improvement 2)
# ================================================================
def active_sampling_query(model, rng, candidate_points, n_samples=1):
    """
    Decide where to sample next using a sequential greedy approach.
    Maximizes the predictive variance while ensuring a space-filling distribution
    by removing nearby candidate points (within 10% of the candidate space range)
    after each selection.
    """
    remaining = np.array(candidate_points).copy()
    selected = []

    # Calculate the exclusion radius dynamically based on candidate space range
    if len(remaining) > 0:
        min_dist = (np.max(remaining) - np.min(remaining)) * 0.10
    else:
        min_dist = 0.5

    for _ in range(n_samples):
        if len(remaining) == 0:
            break
        # Run simulator on remaining points
        y_sim = internal_simulator(rng, model, remaining, n_iter=500)
        variances = np.var(y_sim, axis=1)

        # Pick the point with the highest predictive variance
        best_idx = np.argmax(variances)
        chosen_val = remaining[best_idx]
        selected.append(chosen_val)

        # Space-filling: remove points within the exclusion radius
        mask = np.abs(remaining - chosen_val) >= min_dist
        remaining = remaining[mask]

    return np.array(selected)

# ================================================================
# Evolutionary Engine
# ================================================================
def mutate_model(model, rng, p_step=0.3):
    # Randomly add, remove, or mutate an exponent
    new_exponents = model.exponents.copy()
    if rng.random() < 0.2 and len(new_exponents) < 3: # Add a term
        new_exponents = np.append(new_exponents, rng.uniform(0.1, 5.0))
    elif rng.random() < 0.1 and len(new_exponents) > 1: # Remove a term
        idx = rng.integers(0, len(new_exponents))
        new_exponents = np.delete(new_exponents, idx)
    else: # Mutate an existing exponent
        idx = rng.integers(0, len(new_exponents))
        new_exponents[idx] = np.clip(new_exponents[idx] + rng.normal(0, p_step), 0.1, 5.0)
    
    return MultiModalModel(new_exponents)

def estimate_prob_better(rng, model_a, model_b, ref_x, ref_y, B=100):
    """
    Estimate the probability that model_b is better than model_a
    using bootstrap resampling of the reference set.
    """
    n = len(ref_y)

    # Run simulator and predictions once for both models on full ref_x
    y_sim_a = internal_simulator(rng, model_a, ref_x)
    y_sim_b = internal_simulator(rng, model_b, ref_x)

    y_pred_a = model_a(ref_x)
    y_pred_b = model_b(ref_x)

    better_count = 0

    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)

        # Slice for model_a
        ref_y_b = ref_y[idx]
        y_sim_a_b = y_sim_a[idx, :]
        y_pred_a_b = y_pred_a[idx]

        # R2 and Adjusted R2
        ss_res_a = np.sum((ref_y_b - y_pred_a_b)**2)
        ss_tot_a = np.sum((ref_y_b - np.mean(ref_y_b))**2)
        r2_a = float(1 - ss_res_a / ss_tot_a) if ss_tot_a > 0 else 0.0
        p_a = len(model_a.exponents)
        if n > p_a + 1:
            r2_a = float(np.clip(1.0 - (1.0 - r2_a) * (n - 1) / (n - p_a - 1), 0.0, 1.0))
        else:
            r2_a = 0.0

        lo_a, hi_a = np.percentile(y_sim_a_b, [5, 95], axis=1)
        cov_a = float(np.mean((ref_y_b >= lo_a) & (ref_y_b <= hi_a)))
        mer_a = float(np.max(np.abs(ref_y_b - y_pred_a_b) / (np.abs(ref_y_b) + 1e-12)))
        v_a = hybrid_verisimilitude(r2_a, cov_a, mer_a)

        # Slice for model_b
        y_sim_b_b = y_sim_b[idx, :]
        y_pred_b_b = y_pred_b[idx]

        # R2 and Adjusted R2
        ss_res_b = np.sum((ref_y_b - y_pred_b_b)**2)
        ss_tot_b = np.sum((ref_y_b - np.mean(ref_y_b))**2)
        r2_b = float(1 - ss_res_b / ss_tot_b) if ss_tot_b > 0 else 0.0
        p_b = len(model_b.exponents)
        if n > p_b + 1:
            r2_b = float(np.clip(1.0 - (1.0 - r2_b) * (n - 1) / (n - p_b - 1), 0.0, 1.0))
        else:
            r2_b = 0.0

        lo_b, hi_b = np.percentile(y_sim_b_b, [5, 95], axis=1)
        cov_b = float(np.mean((ref_y_b >= lo_b) & (ref_y_b <= hi_b)))
        mer_b = float(np.max(np.abs(ref_y_b - y_pred_b_b) / (np.abs(ref_y_b) + 1e-12)))
        v_b = hybrid_verisimilitude(r2_b, cov_b, mer_b)

        if v_b > v_a:
            better_count += 1

    return better_count / B

def run_experiment():
    rng = np.random.default_rng(42)
    print("=== General Hypothesis-Testing Engine (v4) ===")
    
    # 1. Initial Data
    calib_x, calib_y = query_external_data(rng, 10, noise_type="heavy_tailed")
    ref_x, ref_y = query_external_data(rng, 30, noise_type="heavy_tailed")
    
    # 2. Evolutionary Search for Structure
    best_model = MultiModalModel([1.0]) # Start with linear
    best_model.fit(calib_x, calib_y)
    
    y_sim = internal_simulator(rng, best_model, ref_x)
    r2 = r2_verisimilitude(best_model, ref_x, ref_y)
    cov = coverage_verisimilitude(y_sim, ref_y)
    mer = max_error_ratio(best_model, ref_x, ref_y)
    best_v = hybrid_verisimilitude(r2, cov, mer)
    
    print(f"Initial Model [p={best_model.exponents}]: V={best_v:.3f} (R2={r2:.3f}, Cov={cov:.3f})")
    
    for gen in range(50):
        # Refresh reference set every generation to prevent overfitting
        ref_x, ref_y = query_external_data(rng, 30, noise_type="heavy_tailed")

        # Propose mutation
        candidate = mutate_model(best_model, rng)
        candidate.fit(calib_x, calib_y)
        
        # Estimate the probability that the candidate is better than best_model
        prob_better = estimate_prob_better(rng, best_model, candidate, ref_x, ref_y, B=100)
        
        if prob_better > 0.55:
            best_model = candidate
            # Point-estimate evaluation of the promoted model for logging
            y_sim_best = internal_simulator(rng, best_model, ref_x)
            r2_best = r2_verisimilitude(best_model, ref_x, ref_y)
            cov_best = coverage_verisimilitude(y_sim_best, ref_y)
            mer_best = max_error_ratio(best_model, ref_x, ref_y)
            best_v = hybrid_verisimilitude(r2_best, cov_best, mer_best)
            print(f"Gen {gen+1}: New Best Promoted [p={best_model.exponents.round(2)}]: V={best_v:.3f} (prob={prob_better:.2f}, R2={r2_best:.3f}, Cov={cov_best:.3f})")

    # 3. Layer 4: Active Data Acquisition
    print("\n--- Layer 4: Active Sampling ---")
    candidate_space = np.linspace(0.5, 10.0, 100)
    next_x = active_sampling_query(best_model, rng, candidate_space, n_samples=5)
    print(f"System decided to sample at x = {next_x.round(2)} (Highest Uncertainty)")
    
    # Simulate acquiring that data
    y_true_next = ground_truth(next_x)
    noise_next = rng.normal(0, 0.10 * y_true_next)
    next_y = y_true_next + noise_next
    
    # Update model with new data
    new_calib_x = np.append(calib_x, next_x)
    new_calib_y = np.append(calib_y, next_y)
    best_model.fit(new_calib_x, new_calib_y)
    
    # Final Evaluation on a broader range
    test_x, test_y = query_external_data(rng, 50, x_range=(0.5, 10.0))
    y_sim = internal_simulator(rng, best_model, test_x)
    r2 = r2_verisimilitude(best_model, test_x, test_y)
    cov = coverage_verisimilitude(y_sim, test_y)
    v = hybrid_verisimilitude(r2, cov, max_error_ratio(best_model, test_x, test_y))
    
    print(f"\nFinal Model [p={best_model.exponents.round(2)}]: V={v:.3f}")
    print(f"Extrapolation Performance (up to x=10): R2={r2:.3f}, Cov={cov:.3f}")

if __name__ == "__main__":
    run_experiment()
