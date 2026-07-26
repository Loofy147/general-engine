import numpy as np
import pandas as pd
import warnings
# Suppress annoying convergence warnings from scikit-learn's robust regressors
warnings.simplefilter("ignore")

from engine.substrate import KaggleDataSource, ConformalCalibrator
from engine.core import MultiModalModel, MultivariateFunctionalTerm, internal_simulator, compute_crps, r2_verisimilitude, coverage_verisimilitude, max_error_ratio
from engine.sandbox import EvolutionarySandbox
from engine.curiosity import CuriosityController
from engine.active_sampling import active_sampling_query

def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("Step 1: Loading Low Alloy Steel Mechanical Properties dataset via KaggleDataSource...")
    filepath = "data/pmo.xlsx"
    target_col = "Tensile Strength (MPa)"
    feature_cols = [
        "C", "Si", "Mn", "Ni", "Cr", "Mo", "Temperature (Â°C)"
    ]

    ds = KaggleDataSource(filepath, target_col, feature_cols)
    X, y = ds.load_data()

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"Features: {feature_cols}")

    # Scale both X and y to [0.1, 1.0] for physical stability
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(), y.max()

    X_scaled = 0.1 + 0.9 * (X - X_min) / (X_max - X_min + 1e-9)
    y_scaled = 0.1 + 0.9 * (y - y_min) / (y_max - y_min + 1e-9)

    # Split the dataset:
    # 100 samples for initial calibration
    # 300 samples for the reference validation set (evolution)
    # The rest for the active oracle pool
    indices = rng.permutation(len(X_scaled))

    cal_idx = indices[:100]
    ref_idx = indices[100:400]
    oracle_idx = indices[400:]

    X_cal, y_cal = X_scaled[cal_idx], y_scaled[cal_idx]
    X_ref, y_ref = X_scaled[ref_idx], y_scaled[ref_idx]
    X_oracle_pool, y_oracle_pool = X_scaled[oracle_idx], y_scaled[oracle_idx]

    print(f"Initial calibration set: {X_cal.shape[0]} samples.")
    print(f"Reference validation set: {X_ref.shape[0]} samples.")
    print(f"Oracle / Active pool: {X_oracle_pool.shape[0]} samples.")

    # 2. Evolutionary Search for Multivariate Physical Laws
    print("\n" + "=" * 80)
    print("Step 2: Running Evolutionary Sandbox to discover steel physical laws...")

    # Set up initial simple linear model on Temperature (feature 6)
    # y = k0 * x6^1.0
    initial_model = MultiModalModel([
        MultivariateFunctionalTerm([[6, "power", 1.0]])
    ])

    curiosity = CuriosityController(window_size=10, initial_success_rate=0.5, rng=rng)
    sandbox = EvolutionarySandbox(max_generations=50, B=50, complexity_method="BIC")

    # Simulates continuous data queries by bootstrapping reference set
    def ref_data_generator(r, size):
        idx = r.choice(len(X_ref), size=size, replace=True)
        return X_ref[idx], y_ref[idx]

    best_model, history = sandbox.run(rng, X_cal, y_cal, initial_model, curiosity, ref_data_generator)

    print("\nDiscovery Complete!")
    print(f"Best Discovered Model: {best_model.terms}")
    print(f"Fitted coefficients (k): {best_model.k}")

    # 3. Conformal Prediction Calibration via Held-Out Oracle
    print("\n" + "=" * 80)
    print("Step 3: Performing Conformal Calibration using held-out Oracle points...")

    # We will pick 10 random points from oracle pool as our calibration reference
    oracle_indices = rng.choice(len(X_oracle_pool), size=10, replace=False)
    X_cal_oracle = X_oracle_pool[oracle_indices]
    y_cal_oracle = y_oracle_pool[oracle_indices]

    # Remove these calibration points from the oracle pool
    mask = np.ones(len(X_oracle_pool), dtype=bool)
    mask[oracle_indices] = False
    X_oracle_pool = X_oracle_pool[mask]
    y_oracle_pool = y_oracle_pool[mask]

    calibrator = ConformalCalibrator(target_coverage=0.90)
    for x_pt, y_pt in zip(X_cal_oracle, y_cal_oracle):
        calibrator.register_calibration_point(x_pt, y_pt)

    calibrator.update_calibration(best_model, rng)
    print(f"Conformal Scaling Factor (q_scale): {calibrator.q_scale:.4f}")

    # 4. Active Sampling Loop
    print("\n" + "=" * 80)
    print("Step 4: Running Active sampling query to reduce uncertainty...")

    # Select 5 most informative points from the oracle pool
    selected_indices = active_sampling_query(best_model, rng, X_oracle_pool, n_samples=5)

    # Find matching indices in the pool
    selected_pts = []
    for pt in selected_indices:
        dists = np.linalg.norm(X_oracle_pool - pt, axis=1)
        idx = np.argmin(dists)
        selected_pts.append(idx)

    print(f"Selected {len(selected_pts)} highly informative candidates from the pool.")

    # Acquire ground-truth measurements
    X_new_measurements = X_oracle_pool[selected_pts]
    y_new_measurements = y_oracle_pool[selected_pts]

    # Remove selected from pool
    mask = np.ones(len(X_oracle_pool), dtype=bool)
    mask[selected_pts] = False
    X_oracle_pool = X_oracle_pool[mask]
    y_oracle_pool = y_oracle_pool[mask]

    # Add new measurements to our calibration set and refit
    X_cal_updated = np.vstack([X_cal, X_new_measurements])
    y_cal_updated = np.append(y_cal, y_new_measurements)

    print("Refitting best model on updated calibration dataset...")
    best_model.fit(X_cal_updated, y_cal_updated, rng=rng)
    print(f"Updated coefficients: {best_model.k}")

    # 5. Final Evaluation and Diagnostics on remaining Pool
    print("\n" + "=" * 80)
    print("Step 5: Evaluating model performance and UQ...")

    y_pred = best_model(X_oracle_pool)
    y_sim = internal_simulator(rng, best_model, X_oracle_pool)
    y_sim_calibrated = calibrator.calibrate_interval(y_pred, y_sim)

    # Diagnostics
    r2 = r2_verisimilitude(best_model, X_oracle_pool, y_oracle_pool)
    coverage_raw = coverage_verisimilitude(y_sim, y_oracle_pool)
    coverage_calibrated = coverage_verisimilitude(y_sim_calibrated, y_oracle_pool)
    crps_raw = compute_crps(y_sim, y_oracle_pool)
    crps_calibrated = compute_crps(y_sim_calibrated, y_oracle_pool)

    print("-" * 50)
    print("                 STEEL DISCOVERY REPORT & STATISTICS")
    print("-" * 50)
    print(f"Evolved Formula Terms:  {best_model.terms}")
    print(f"Fitted k-coefficients:  {best_model.k.round(4)}")
    print(f"R^2 Score on Oracle Set: {r2:.4f}")
    print(f"Raw CRPS:               {crps_raw:.4f}")
    print(f"Calibrated CRPS:        {crps_calibrated:.4f}")
    print(f"Uncalibrated 90% Cov:   {coverage_raw * 100:.2f}%")
    print(f"Conformalized 90% Cov:  {coverage_calibrated * 100:.2f}% (Target: 90.00%)")
    print("-" * 50)
    print("Self-Proving Steel Loop Closed Successfully!")

if __name__ == "__main__":
    main()
