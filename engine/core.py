import numpy as np
from sklearn.linear_model import HuberRegressor

class MultiModalModel:
    """
    A model representing y = sum(k_i * x^p_i).
    """
    def __init__(self, exponents):
        self.exponents = np.array(exponents, dtype=float)
        self.k = np.zeros(len(exponents))
        self.k_se = np.zeros(len(exponents))
        self.cov_matrix = None
        self.residual_scale = 0.10

    def __call__(self, x, k=None):
        if k is None:
            k = self.k
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

def internal_simulator(rng, model, x_query, n_iter=2000):
    """
    Simulates y values using multivariate normal parameters.
    """
    if model.cov_matrix is not None:
        # Sample k from the multivariate normal distribution
        try:
            k_samples = rng.multivariate_normal(model.k, model.cov_matrix, size=n_iter).T
        except Exception:
            # Fallback if covariance matrix is not positive semi-definite
            k_samples = rng.normal(model.k[:, None], np.abs(model.k[:, None]) * 0.1 + 1e-9, size=(len(model.k), n_iter))
    else:
        # Fallback to simple relative uncertainty if covariance is missing
        k_samples = rng.normal(model.k[:, None], np.abs(model.k[:, None]) * 0.1 + 1e-9, size=(len(model.k), n_iter))

    y_pred = model(x_query, k=k_samples)
    res_scale = getattr(model, 'residual_scale', 0.10)
    noise = rng.normal(0, res_scale * np.abs(y_pred) + 1e-9)
    return y_pred + noise

def coverage_verisimilitude(y_sim, ref_y):
    """
    Fraction of reference points whose true value falls inside
    the model's 90% predictive interval (5th to 95th percentile).
    """
    lo, hi = np.percentile(y_sim, [5, 95], axis=1)
    return float(np.mean((ref_y >= lo) & (ref_y <= hi)))

def r2_verisimilitude(model, ref_x, ref_y):
    """
    Adjusted R2 to penalize complexity (number of terms/exponents).
    """
    y_pred = model(ref_x)
    ss_res = np.sum((ref_y - y_pred)**2)
    ss_tot = np.sum((ref_y - np.mean(ref_y))**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    n = len(ref_y)
    p = len(model.exponents)
    if n > p + 1:
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = 0.0
    return float(np.clip(adj_r2, 0.0, 1.0))

def max_error_ratio(model, ref_x, ref_y, eps=1e-12):
    """
    Max relative error across reference set to penalize catastrophic failures.
    """
    y_pred = model(ref_x)
    return float(np.max(np.abs(ref_y - y_pred) / (np.abs(ref_y) + eps)))

def compute_aic(model, ref_x, ref_y, eps=1e-12):
    """
    Akaike Information Criterion (AIC) approximated via MSE.
    """
    y_pred = model(ref_x)
    mse = np.mean((ref_y - y_pred)**2)
    p = len(model.exponents)
    n = len(ref_y)
    return float(n * np.log(max(mse, eps)) + 2 * p)

def compute_bic(model, ref_x, ref_y, eps=1e-12):
    """
    Bayesian Information Criterion (BIC) approximated via MSE.
    """
    y_pred = model(ref_x)
    mse = np.mean((ref_y - y_pred)**2)
    p = len(model.exponents)
    n = len(ref_y)
    return float(n * np.log(max(mse, eps)) + p * np.log(n))

def get_complexity_penalty(model, n, method="BIC"):
    """
    Complexity penalty normalized to scale with the hybrid score.
    BIC: (p * ln(n)) / (2 * n)
    AIC: (2 * p) / (2 * n)
    """
    p = len(model.exponents)
    if method == "BIC":
        return (p * np.log(n)) / (2.0 * n) if n > 0 else 0.0
    elif method == "AIC":
        return (2.0 * p) / (2.0 * n) if n > 0 else 0.0
    return 0.0

def hybrid_verisimilitude(r2, cov, mer, lam=0.25, complexity_penalty=0.0):
    """
    Hybrid metric combining accuracy, predictive calibration, worst-case error,
    and complexity penalty.
    """
    r2_eff = max(0.0, r2)
    harmonic = (2.0 * r2_eff * cov / (r2_eff + cov)) if (r2_eff + cov) > 0 else 0.0
    score = harmonic - lam * mer - complexity_penalty
    return float(np.clip(score, 0.0, 1.0))

def classify_model(hybrid_score, theta_promote=0.85):
    """
    Classify model as PROMOTED, INHIBITED, or GAP.
    """
    if hybrid_score > theta_promote:
        return "PROMOTED"
    elif hybrid_score < 0.3:
        return "INHIBITED"
    else:
        return "GAP"
