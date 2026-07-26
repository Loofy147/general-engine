import numpy as np
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor

class FunctionalTerm:
    def __init__(self, term_type="power", p=1.0):
        self.term_type = term_type  # "power", "exp", "log"
        self.p = float(p)

    def evaluate(self, x):
        x = np.atleast_1d(x)
        if self.term_type == "power":
            return np.power(np.abs(x), self.p)
        elif self.term_type == "exp":
            return np.exp(-self.p * np.abs(x))
        elif self.term_type == "log":
            return np.log(1.0 + self.p * np.abs(x))
        else:
            raise ValueError(f"Unknown term type: {self.term_type}")

    def copy(self):
        return FunctionalTerm(self.term_type, self.p)

    def __repr__(self):
        return f"{self.term_type}({self.p:.2f})"


class MultiModalModel:
    """
    A model representing y = sum(k_i * f_i(x)).
    """
    def __init__(self, terms_or_exponents):
        self.terms = []
        if isinstance(terms_or_exponents, (list, tuple, np.ndarray)):
            for item in terms_or_exponents:
                if isinstance(item, FunctionalTerm):
                    self.terms.append(item.copy())
                elif isinstance(item, (int, float, np.floating, np.integer)):
                    self.terms.append(FunctionalTerm("power", item))
                elif isinstance(item, tuple) and len(item) == 2:
                    self.terms.append(FunctionalTerm(item[0], item[1]))
                else:
                    raise ValueError(f"Invalid term specifier: {item}")
        else:
            if isinstance(terms_or_exponents, FunctionalTerm):
                self.terms.append(terms_or_exponents.copy())
            else:
                self.terms.append(FunctionalTerm("power", terms_or_exponents))

        self.k = np.zeros(len(self.terms))
        self.k_se = np.zeros(len(self.terms))
        self.cov_matrix = None
        self.k_bootstrap = None
        self.residual_scale = 0.10

    @property
    def exponents(self):
        return np.array([t.p for t in self.terms], dtype=float)

    @exponents.setter
    def exponents(self, value):
        self.terms = [FunctionalTerm("power", p) for p in value]
        self.k = np.zeros(len(self.terms))
        self.k_se = np.zeros(len(self.terms))
        self.k_bootstrap = None

    def get_design_matrix(self, x):
        x = np.atleast_1d(x)
        cols = [t.evaluate(x) for t in self.terms]
        return np.column_stack(cols)

    def __call__(self, x, k=None):
        if k is None:
            k = self.k
        x = np.atleast_1d(x)
        design_matrix = self.get_design_matrix(x)

        if k.ndim == 1:
            # Standard prediction: sum(k_i * f_i(x))
            # Result shape: (N,)
            return design_matrix @ k
        else:
            # Simulator prediction with M samples of k
            # k shape: (len(terms), M)
            # Result shape: (N, M)
            return design_matrix @ k

    def fit(self, x, y, B=100, rng=None):
        """Robust regression for k given fixed exponents using RANSAC + Theil-Sen."""
        if rng is None:
            rng = np.random.default_rng(42)
        X = self.get_design_matrix(x)
        n, m = X.shape

        # 1. Main robust fit using RANSAC with Theil-Sen
        try:
            base = TheilSenRegressor(fit_intercept=False, random_state=42)
            min_samples = max(m + 1, min(5, n))
            if min_samples > n:
                min_samples = n
            ransac = RANSACRegressor(estimator=base, min_samples=min_samples, random_state=42)
            ransac.fit(X, y)
            self.k = ransac.estimator_.coef_
            inliers = ransac.inlier_mask_
        except Exception:
            try:
                ts = TheilSenRegressor(fit_intercept=False, random_state=42)
                ts.fit(X, y)
                self.k = ts.coef_
                inliers = np.ones(n, dtype=bool)
            except Exception:
                try:
                    self.k, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    inliers = np.ones(n, dtype=bool)
                except Exception:
                    self.k = np.zeros(m)
                    inliers = np.ones(n, dtype=bool)

        # Approximate a simple diagonal covariance matrix for backward compatibility
        try:
            y_pred = X @ self.k
            mse = np.sum((y - y_pred)**2) / max(n - m, 1)
            self.cov_matrix = mse * np.linalg.inv(X.T @ X + 1e-6 * np.eye(m))
            self.k_se = np.sqrt(np.diag(self.cov_matrix))
        except Exception:
            self.cov_matrix = np.eye(m) * 0.1
            self.k_se = np.ones(m) * 0.1

        # 2. Non-parametric bootstrap of the RANSAC fitting process
        k_boot_list = []
        for _ in range(B):
            boot_idx = rng.choice(n, size=n, replace=True)
            X_b = X[boot_idx]
            y_b = y[boot_idx]
            try:
                base_b = TheilSenRegressor(fit_intercept=False, random_state=rng.integers(0, 100000))
                min_s = max(m + 1, min(5, n))
                if min_s > n:
                    min_s = n
                ransac_b = RANSACRegressor(estimator=base_b, min_samples=min_s, random_state=rng.integers(0, 100000))
                ransac_b.fit(X_b, y_b)
                k_b = ransac_b.estimator_.coef_
            except Exception:
                try:
                    ts_b = TheilSenRegressor(fit_intercept=False, random_state=rng.integers(0, 100000))
                    ts_b.fit(X_b, y_b)
                    k_b = ts_b.coef_
                except Exception:
                    try:
                        k_b, _, _, _ = np.linalg.lstsq(X_b, y_b, rcond=None)
                    except Exception:
                        k_b = self.k.copy()
            k_boot_list.append(k_b)

        self.k_bootstrap = np.column_stack(k_boot_list)  # Shape (m, B)

        # Estimate residual scale
        y_pred = X @ self.k
        if np.any(inliers):
            residuals = y[inliers] - y_pred[inliers]
        else:
            residuals = y - y_pred
        mad = np.median(np.abs(residuals - np.median(residuals)))
        self.residual_scale = float(max(mad * 1.4826, 1e-4))


def internal_simulator(rng, model, x_query, n_iter=2000):
    """
    Simulates y values using bootstrap parameter distributions.
    """
    if getattr(model, 'k_bootstrap', None) is not None:
        num_boot = model.k_bootstrap.shape[1]
        boot_idx = rng.choice(num_boot, size=n_iter, replace=True)
        k_samples = model.k_bootstrap[:, boot_idx]  # Shape (m, n_iter)
    else:
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
    p = len(model.terms)
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
    p = len(model.terms)
    n = len(ref_y)
    return float(n * np.log(max(mse, eps)) + 2 * p)


def compute_bic(model, ref_x, ref_y, eps=1e-12):
    """
    Bayesian Information Criterion (BIC) approximated via MSE.
    """
    y_pred = model(ref_x)
    mse = np.mean((ref_y - y_pred)**2)
    p = len(model.terms)
    n = len(ref_y)
    return float(n * np.log(max(mse, eps)) + p * np.log(n))


def get_complexity_penalty(model, n, method="BIC"):
    """
    Complexity penalty normalized to scale with the hybrid score.
    BIC: (p * ln(n)) / (2 * n)
    AIC: (2 * p) / (2 * n)
    """
    p = len(model.terms)
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


def classify_model(hybrid_score, theta_promote=0.85, y_sim=None, ref_y=None):
    """
    Classify model as PROMOTED, INHIBITED, or GAP.
    If y_sim and ref_y are provided, additional calibration tests must pass for promotion:
    1. Bayesian coverage test (lower credible bound of coverage must be high)
    2. PIT calibration test (Kolmogorov-Smirnov test for quantile uniformity)
    """
    if hybrid_score > theta_promote:
        if y_sim is not None and ref_y is not None:
            coverage_ok, _, _ = check_bayesian_coverage(y_sim, ref_y, target_coverage=0.80, alpha_threshold=0.95)
            pit_ok, _ = check_pit_calibration(y_sim, ref_y, alpha=0.05)
            if coverage_ok and pit_ok:
                return "PROMOTED"
            else:
                return "GAP"
        return "PROMOTED"
    elif hybrid_score < 0.3:
        return "INHIBITED"
    else:
        return "GAP"

def compute_crps(y_samples, y_true):
    """
    Computes the Continuous Ranked Probability Score (CRPS)
    between simulated samples and the true observations.
    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    We use an efficient sorting-based algorithm:
    CRPS = 1/M * sum_{i=1}^M |x_i - y| - 1/(M^2) * sum_{i=1}^M (2i - M - 1) * x_{(i)}
    where x_{(1)} <= ... <= x_{(M)} are sorted samples.
    """
    y_samples = np.atleast_2d(y_samples)
    y_true = np.atleast_1d(y_true)
    N, M = y_samples.shape

    # Sort samples for each reference point
    z = np.sort(y_samples, axis=1)

    # Term 1: E|X - y| = 1/M * sum |z_i - y|
    term1 = np.mean(np.abs(z - y_true[:, None]), axis=1)

    # Term 2: 0.5 * E|X - X'| = 1/(M^2) * sum (2i - M - 1) * z_i
    # Note: Using 1-based indexing for formula: i in [1, M] -> (2*i - M - 1)
    # Using 0-based indexing: k in [0, M-1] -> (2*(k+1) - M - 1) = (2k + 1 - M)
    k = np.arange(M)
    weights = 2 * k + 1 - M
    term2 = np.sum(z * weights[None, :], axis=1) / (M ** 2)

    crps_per_point = term1 - term2
    return float(np.mean(crps_per_point))
from scipy.stats import beta, kstest

def check_bayesian_coverage(y_sim, ref_y, target_coverage=0.80, alpha_threshold=0.95):
    """
    Computes empirical coverage (fraction of reference points falling within the 90%
    prediction interval) and performs a Bayesian binomial posterior check.
    Returns (success, lower_credible_bound, empirical_coverage)
    """
    # 1. Compute empirical coverage on 90% prediction intervals
    lo, hi = np.percentile(y_sim, [5, 95], axis=1)
    successes = np.sum((ref_y >= lo) & (ref_y <= hi))
    n = len(ref_y)

    # 2. Bayesian binomial posterior with conjugate Beta prior (Jeffreys prior: Beta(0.5, 0.5))
    a_post = 0.5 + successes
    b_post = 0.5 + n - successes

    # Probability that coverage > target_coverage
    prob_above_target = 1.0 - beta.cdf(target_coverage, a_post, b_post)
    empirical_coverage = float(successes / n) if n > 0 else 0.0

    # Return True if the posterior probability is above the confidence alpha_threshold
    # e.g., P(Coverage > 0.80) > 0.95
    success = bool(prob_above_target >= (1.0 - alpha_threshold))
    return success, prob_above_target, empirical_coverage


def check_pit_calibration(y_sim, ref_y, alpha=0.05):
    """
    Probability Integral Transform (PIT) calibration check using the Kolmogorov-Smirnov test.
    For each y_i, computes its CDF value under the simulated distribution.
    Under perfect calibration, these PIT values are uniformly distributed on [0, 1].
    Returns (passes_ks, p_value)
    """
    # y_sim shape: (N, M) where N = len(ref_y), M = n_iter
    N, M = y_sim.shape
    pit_values = []

    for i in range(N):
        y_val = ref_y[i]
        sims = y_sim[i]
        # Empirical CDF value
        pit_val = np.mean(sims <= y_val)
        # Avoid exact 0 or 1 boundaries slightly to keep continuous and clean
        pit_val = np.clip(pit_val, 1e-9, 1.0 - 1e-9)
        pit_values.append(pit_val)

    pit_values = np.array(pit_values)

    # One-sample KS test against a uniform distribution
    res = kstest(pit_values, 'uniform')
    p_value = float(res.pvalue)

    # passes_ks is True if we fail to reject the null hypothesis of uniformity
    passes_ks = bool(p_value >= alpha)
    return passes_ks, p_value
