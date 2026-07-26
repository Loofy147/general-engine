import numpy as np
from engine.core import MultiModalModel, internal_simulator, r2_verisimilitude, coverage_verisimilitude, max_error_ratio, hybrid_verisimilitude
from engine.substrate import MetaLoopSubstrate, RoutingPolicy

# ================================================================
# Enzyme Kinetics Ground Truth & Data Generation
# ================================================================
# Vmax = 10.0, Km = 2.0
V_MAX = 10.0
K_M = 2.0

def enzyme_ground_truth(x):
    """
    Michaelis-Menten Enzyme Kinetics reaction rate:
    v = (Vmax * x) / (Km + x)
    """
    return (V_MAX * x) / (K_M + x)

def query_enzyme_data(rng, n, x_range=(0.2, 10.0), noise_type="heavy_tailed"):
    """
    Generates experimental data for enzyme kinetics with realistic noise.
    - 'heteroscedastic': noise scales with reaction velocity
    - 'heavy_tailed': normal noise mixed with Cauchy outliers
    """
    x = rng.uniform(*x_range, size=n)
    y_true = enzyme_ground_truth(x)

    if noise_type == "heteroscedastic":
        # noise standard deviation scales with reaction velocity
        noise = rng.normal(0, 0.12 * y_true + 1e-9)
    elif noise_type == "heavy_tailed":
        # 90% normal noise, 10% Cauchy-like outlier noise
        noise = rng.normal(0, 0.08 * y_true + 1e-9) + rng.standard_cauchy(size=n) * 0.03 * y_true
    else:
        noise = rng.normal(0, 0.05 * y_true + 1e-9)

    # Reaction velocity cannot be negative in physical experiments
    y_measured = np.clip(y_true + noise, 0.0, None)
    return x, y_measured


class EnzymeKineticsPilot:
    """
    Coordinates the Enzyme Kinetics simulated laboratory pilot.
    """
    def __init__(self, complexity_method="BIC"):
        self.complexity_method = complexity_method
        # Initialize the substrate with a blind linear hypothesis: y = k * x^1.0
        self.substrate = MetaLoopSubstrate(
            ground_truth_fn=enzyme_ground_truth,
            initial_model=MultiModalModel([1.0]),
            complexity_method=complexity_method
        )

    def run_pilot(self, rng):
        logs = []
        logs.append("Initializing Enzyme Kinetics Pilot Substrate...")
        logs.append(f"Ground Truth Process: Michaelis-Menten [Vmax={V_MAX}, Km={K_M}]")

        # 1. Warm-up: collect initial data and fit base model
        logs.append("\n--- Phase 1: Initial Calibration ---")
        cal_x, cal_y = query_enzyme_data(rng, 10, noise_type="heavy_tailed")
        self.substrate.knowledge.model.fit(cal_x, cal_y)
        logs.append(f"Initial model exponents: {self.substrate.knowledge.model.exponents}")
        logs.append(f"Initial fitted scale parameters (k): {self.substrate.knowledge.model.k}")

        # Define data generator helper for the sandbox
        def data_gen(r, size):
            return query_enzyme_data(r, size, noise_type="heavy_tailed")

        # 2. Run scientific model evolution
        logs.append("\n--- Phase 2: Scientific Hypothesis Search (Evolution) ---")
        prev_exponents = self.substrate.knowledge.model.exponents.copy()
        model, evolution_history = self.substrate.knowledge.update_scientific_model(
            rng, cal_x, cal_y, data_gen
        )
        logs.append(f"Evolution complete. Best structural model exponents: {model.exponents.round(2)}")
        logs.append(f"Fitted coefficients (k): {model.k.round(4)}")

        # Log evolution improvements
        for step in evolution_history:
            logs.append(
                f"Gen {step['generation']}: Model exponents: {step['model_exponents'].round(2)} | "
                f"Score: {step['score']:.4f} | Status: {step['status']} | Theta_promote: {step['theta_promote']:.3f}"
            )

        # 3. Simulate processing workflow of requests and generating audited results
        logs.append("\n--- Phase 3: Workflow Execution & Audit Plane Logging ---")
        for i in range(15):
            # Simulated incoming request
            raw_req = {
                "source": "lab_scheduler",
                "event_type": "measurement_request",
                "data": {
                    "x": float(rng.uniform(0.5, 9.0))
                }
            }
            res = self.substrate.process_measurement_request(rng, raw_req)
            logs.append(
                f"Request {i+1}: concentration x={raw_req['data']['x']:.2f} -> "
                f"routed to {res['data']['sensor_used']} -> measured rate y={res['data']['y']:.3f} (cost={res['data']['cost']})"
            )

        # Get performance report
        perf_report = self.substrate.audit.get_performance_report()
        logs.append(f"\nAudit Performance Report: {perf_report}")

        # 4. Meta-Loop: Policy optimization
        logs.append("\n--- Phase 4: Policy Self-Experimentation & Meta-Loop ---")
        logs.append(f"Current routing threshold: {self.substrate.active_policy.threshold:.3f}")
        best_score, opt_log = self.substrate.self_experiment_and_optimize_policies(rng, n_mutations=15)
        logs.append(f"Policy Optimization complete. Evolved routing threshold: {self.substrate.active_policy.threshold:.3f}")
        for o in opt_log:
            logs.append(
                f"Step {o['step']}: Threshold {o['old_threshold']:.2f} -> {o['new_threshold']:.2f} | "
                f"Score improved to {o['new_score']:.4f}"
            )

        # 5. Active sampling to reduce uncertainty in weak areas
        logs.append("\n--- Phase 5: Active Data Acquisition ---")
        candidate_concentrations = np.linspace(0.2, 10.0, 100)
        next_xs = self.substrate.knowledge.model.exponents
        from engine.active_sampling import active_sampling_query
        target_xs = active_sampling_query(self.substrate.knowledge.model, rng, candidate_concentrations, n_samples=3)
        logs.append(f"Substrate targeted concentration points for next run: {target_xs.round(3)}")

        return logs
