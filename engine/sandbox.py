import numpy as np
from engine.core import (
    MultiModalModel, internal_simulator, r2_verisimilitude,
    coverage_verisimilitude, max_error_ratio, hybrid_verisimilitude,
    get_complexity_penalty, classify_model
)

def mutate_model(model, rng, p_step=0.3):
    """
    Randomly add, remove, or mutate an exponent in the sum of power laws.
    """
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

def estimate_prob_better(rng, model_a, model_b, ref_x, ref_y, B=100, complexity_method="BIC"):
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

    # Calculate complexity penalties based on the chosen method
    penalty_a = get_complexity_penalty(model_a, n, method=complexity_method)
    penalty_b = get_complexity_penalty(model_b, n, method=complexity_method)

    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)

        # Slice for model_a
        ref_y_resample = ref_y[idx]
        y_sim_a_resample = y_sim_a[idx, :]
        y_pred_a_resample = y_pred_a[idx]

        # R2 and Adjusted R2
        ss_res_a = np.sum((ref_y_resample - y_pred_a_resample)**2)
        ss_tot_a = np.sum((ref_y_resample - np.mean(ref_y_resample))**2)
        r2_a = float(1 - ss_res_a / ss_tot_a) if ss_tot_a > 0 else 0.0
        p_a = len(model_a.exponents)
        if n > p_a + 1:
            r2_a = float(np.clip(1.0 - (1.0 - r2_a) * (n - 1) / (n - p_a - 1), 0.0, 1.0))
        else:
            r2_a = 0.0

        lo_a, hi_a = np.percentile(y_sim_a_resample, [5, 95], axis=1)
        cov_a = float(np.mean((ref_y_resample >= lo_a) & (ref_y_resample <= hi_a)))
        mer_a = float(np.max(np.abs(ref_y_resample - y_pred_a_resample) / (np.abs(ref_y_resample) + 1e-12)))
        v_a = hybrid_verisimilitude(r2_a, cov_a, mer_a, complexity_penalty=penalty_a)

        # Slice for model_b
        y_sim_b_resample = y_sim_b[idx, :]
        y_pred_b_resample = y_pred_b[idx]

        # R2 and Adjusted R2
        ss_res_b = np.sum((ref_y_resample - y_pred_b_resample)**2)
        ss_tot_b = np.sum((ref_y_resample - np.mean(ref_y_resample))**2)
        r2_b = float(1 - ss_res_b / ss_tot_b) if ss_tot_b > 0 else 0.0
        p_b = len(model_b.exponents)
        if n > p_b + 1:
            r2_b = float(np.clip(1.0 - (1.0 - r2_b) * (n - 1) / (n - p_b - 1), 0.0, 1.0))
        else:
            r2_b = 0.0

        lo_b, hi_b = np.percentile(y_sim_b_resample, [5, 95], axis=1)
        cov_b = float(np.mean((ref_y_resample >= lo_b) & (ref_y_resample <= hi_b)))
        mer_b = float(np.max(np.abs(ref_y_resample - y_pred_b_resample) / (np.abs(ref_y_resample) + 1e-12)))
        v_b = hybrid_verisimilitude(r2_b, cov_b, mer_b, complexity_penalty=penalty_b)

        if v_b > v_a:
            better_count += 1

    return better_count / B


class EvolutionarySandbox:
    """
    Evolutionary Sandbox that mutates model structural hypotheses,
    selects them based on bootstrap-estimated hybrid verisimilitude,
    and updates adaptive curiosity thresholds.
    """
    def __init__(self, max_generations=50, B=100, complexity_method="BIC"):
        self.max_generations = max_generations
        self.B = B
        self.complexity_method = complexity_method

    def run(self, rng, calib_x, calib_y, initial_model, curiosity_controller, data_generator_fn):
        """
        Runs the evolutionary search.
        data_generator_fn should be a function: (rng, size) -> (ref_x, ref_y)
        """
        best_model = initial_model
        best_model.fit(calib_x, calib_y)

        # Initial evaluation
        ref_x, ref_y = data_generator_fn(rng, 30)
        y_sim = internal_simulator(rng, best_model, ref_x)
        r2 = r2_verisimilitude(best_model, ref_x, ref_y)
        cov = coverage_verisimilitude(y_sim, ref_y)
        mer = max_error_ratio(best_model, ref_x, ref_y)
        penalty = get_complexity_penalty(best_model, len(ref_y), method=self.complexity_method)
        best_v = hybrid_verisimilitude(r2, cov, mer, complexity_penalty=penalty)

        theta_promote = curiosity_controller.get_threshold()
        status = classify_model(best_v, theta_promote)
        curiosity_controller.record_evaluation(status == "PROMOTED")

        history = [{
            "generation": 0,
            "model_exponents": best_model.exponents.copy(),
            "score": best_v,
            "status": status,
            "theta_promote": theta_promote
        }]

        if status == "PROMOTED":
            return best_model, history

        for gen in range(1, self.max_generations + 1):
            # Refresh reference set to prevent overfitting
            ref_x, ref_y = data_generator_fn(rng, 30)

            # Propose and fit mutation
            candidate = mutate_model(best_model, rng)
            candidate.fit(calib_x, calib_y)

            # Bootstrap comparison
            prob_better = estimate_prob_better(
                rng, best_model, candidate, ref_x, ref_y, B=self.B, complexity_method=self.complexity_method
            )

            # We accept the candidate if it is confidently better (e.g., prob > 0.55)
            if prob_better > 0.55:
                best_model = candidate
                y_sim = internal_simulator(rng, best_model, ref_x)
                r2 = r2_verisimilitude(best_model, ref_x, ref_y)
                cov = coverage_verisimilitude(y_sim, ref_y)
                mer = max_error_ratio(best_model, ref_x, ref_y)
                penalty = get_complexity_penalty(best_model, len(ref_y), method=self.complexity_method)
                best_v = hybrid_verisimilitude(r2, cov, mer, complexity_penalty=penalty)

                theta_promote = curiosity_controller.get_threshold()
                status = classify_model(best_v, theta_promote)
                curiosity_controller.record_evaluation(status == "PROMOTED")

                history.append({
                    "generation": gen,
                    "model_exponents": best_model.exponents.copy(),
                    "score": best_v,
                    "status": status,
                    "theta_promote": theta_promote
                })

                # Stop search early if model has been promoted
                if status == "PROMOTED":
                    break

        return best_model, history
