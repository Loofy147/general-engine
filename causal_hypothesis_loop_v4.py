import numpy as np
from scipy.optimize import curve_fit

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
        """Linear least squares for k given fixed exponents."""
        design_matrix = np.power(x[:, None], self.exponents[None, :])
        # Solve (X^T X) k = X^T y
        try:
            k_hat, residuals, rank, s = np.linalg.lstsq(design_matrix, y, rcond=None)
            self.k = k_hat
            
            # Improvement 4: Robust Covariance Estimation
            n, m = design_matrix.shape
            if n > m:
                mse = np.sum((y - design_matrix @ k_hat)**2) / (n - m)
                self.cov_matrix = mse * np.linalg.inv(design_matrix.T @ design_matrix)
                self.k_se = np.sqrt(np.diag(self.cov_matrix))
            else:
                self.k_se = np.zeros(m)
        except np.linalg.LinAlgError:
            self.k = np.zeros(len(self.exponents))
            self.k_se = np.zeros(len(self.exponents))

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
    # Add residual noise estimate (using the MSE from the fit if available)
    noise_level = 0.10 # simplified
    return y_pred + rng.normal(0, noise_level * np.abs(y_pred) + 1e-9)

def coverage_verisimilitude(y_sim, ref_y):
    lo, hi = np.percentile(y_sim, [5, 95], axis=1)
    return float(np.mean((ref_y >= lo) & (ref_y <= hi)))

def r2_verisimilitude(model, ref_x, ref_y):
    y_pred = model(ref_x)
    ss_res = np.sum((ref_y - y_pred)**2)
    ss_tot = np.sum((ref_y - np.mean(ref_y))**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

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
    Decide where to sample next by finding points with maximum predictive variance.
    (Simple version of Optimal Experimental Design)
    """
    # Sample many predictions to see where they disagree most
    y_sim = internal_simulator(rng, model, candidate_points, n_iter=500)
    variances = np.var(y_sim, axis=1)
    # Pick the point with the highest uncertainty
    best_idx = np.argsort(variances)[-n_samples:]
    return candidate_points[best_idx]

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
        # Propose mutation
        candidate = mutate_model(best_model, rng)
        candidate.fit(calib_x, calib_y)
        
        # Evaluate
        y_sim = internal_simulator(rng, candidate, ref_x)
        r2 = r2_verisimilitude(candidate, ref_x, ref_y)
        cov = coverage_verisimilitude(y_sim, ref_y)
        mer = max_error_ratio(candidate, ref_x, ref_y)
        v = hybrid_verisimilitude(r2, cov, mer)
        
        # Slightly favor simpler models (parsimony)
        v -= 0.05 * (len(candidate.exponents) - 1)
        
        if v > best_v:
            best_model, best_v = candidate, v
            print(f"Gen {gen+1}: New Best [p={best_model.exponents.round(2)}]: V={best_v:.3f} (R2={r2:.3f}, Cov={cov:.3f})")

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
