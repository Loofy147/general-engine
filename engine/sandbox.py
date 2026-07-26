import numpy as np
from engine.core import (
    MultiModalModel, FunctionalTerm, internal_simulator,
    classify_model, get_complexity_penalty, compute_crps
)

def mutate_model(model, rng, p_step=0.3):
    """
    Randomly add, remove, or mutate a term (power, exp, log) in the model.
    """
    new_terms = [t.copy() for t in model.terms]

    # 20% chance to add a term (if we have less than 3 terms)
    if rng.random() < 0.2 and len(new_terms) < 3:
        t_type = rng.choice(["power", "exp", "log"])
        p_val = rng.uniform(0.1, 5.0)
        new_terms.append(FunctionalTerm(t_type, p_val))
    # 10% chance to remove a term (if we have more than 1 term)
    elif rng.random() < 0.1 and len(new_terms) > 1:
        idx = rng.integers(0, len(new_terms))
        new_terms.pop(idx)
    # Otherwise mutate an existing term
    else:
        idx = rng.integers(0, len(new_terms))
        term = new_terms[idx]
        # 80% chance to mutate parameter, 20% to change term type
        if rng.random() < 0.8:
            term.p = float(np.clip(term.p + rng.normal(0, p_step), 0.1, 5.0))
        else:
            term.term_type = rng.choice(["power", "exp", "log"])
            term.p = float(rng.uniform(0.1, 5.0))

    return MultiModalModel(new_terms)


def estimate_prob_better(rng, model_a, model_b, ref_x, ref_y, B=100, complexity_method="BIC"):
    """
    Estimate the probability that model_b is better than model_a
    using bootstrap resampling of the reference set and CRPS scoring.
    """
    n = len(ref_y)

    # Run simulator once for both models on full ref_x
    y_sim_a = internal_simulator(rng, model_a, ref_x)
    y_sim_b = internal_simulator(rng, model_b, ref_x)

    better_count = 0

    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)

        ref_y_resample = ref_y[idx]
        y_sim_a_resample = y_sim_a[idx, :]
        y_sim_b_resample = y_sim_b[idx, :]

        crps_a = compute_crps(y_sim_a_resample, ref_y_resample)
        crps_b = compute_crps(y_sim_b_resample, ref_y_resample)

        penalty_a = get_complexity_penalty(model_a, n, method=complexity_method)
        penalty_b = get_complexity_penalty(model_b, n, method=complexity_method)

        # We minimize CRPS + complexity penalty
        score_a = crps_a + penalty_a
        score_b = crps_b + penalty_b

        if score_b < score_a:
            better_count += 1

    return better_count / B


class EvolutionarySandbox:
    """
    Evolutionary Sandbox that mutates model structural hypotheses,
    selects them based on bootstrap-estimated CRPS verisimilitude,
    and updates adaptive curiosity thresholds.
    """
    def __init__(self, max_generations=50, B=100, complexity_method="BIC", restart_stagnation=10):
        self.max_generations = max_generations
        self.B = B
        self.complexity_method = complexity_method
        self.restart_stagnation = restart_stagnation

    def run(self, rng, calib_x, calib_y, initial_model, curiosity_controller, data_generator_fn):
        """
        Runs the evolutionary search with adaptive (1+1)-ES mutation rule and restarts.
        """
        best_model = initial_model
        best_model.fit(calib_x, calib_y, rng=rng)

        # Initial evaluation
        ref_x, ref_y = data_generator_fn(rng, 30)
        y_sim = internal_simulator(rng, best_model, ref_x)
        crps_val = compute_crps(y_sim, ref_y)
        penalty = get_complexity_penalty(best_model, len(ref_y), method=self.complexity_method)

        # Define score in [0, 1] as exp(-crps)
        best_score = float(np.exp(-(crps_val + penalty)))

        theta_promote = curiosity_controller.get_threshold()
        status = classify_model(best_score, theta_promote, y_sim, ref_y)
        curiosity_controller.record_evaluation(status == "PROMOTED")

        history = [{
            "generation": 0,
            "model_exponents": best_model.exponents.copy(),
            "score": best_score,
            "status": status,
            "theta_promote": theta_promote
        }]

        if status == "PROMOTED":
            return best_model, history

        sigma = 0.3
        stagnation_counter = 0

        for gen in range(1, self.max_generations + 1):
            # Refresh reference set to prevent overfitting
            ref_x, ref_y = data_generator_fn(rng, 30)

            # Propose and fit mutation
            candidate = mutate_model(best_model, rng, p_step=sigma)
            candidate.fit(calib_x, calib_y, rng=rng)

            # Bootstrap comparison
            prob_better = estimate_prob_better(
                rng, best_model, candidate, ref_x, ref_y, B=self.B, complexity_method=self.complexity_method
            )

            # We accept the candidate if it is confidently better (e.g., prob > 0.55)
            if prob_better > 0.55:
                best_model = candidate
                y_sim = internal_simulator(rng, best_model, ref_x)
                crps_val = compute_crps(y_sim, ref_y)
                penalty = get_complexity_penalty(best_model, len(ref_y), method=self.complexity_method)
                best_score = float(np.exp(-(crps_val + penalty)))

                theta_promote = curiosity_controller.get_threshold()
                status = classify_model(best_score, theta_promote, y_sim, ref_y)
                curiosity_controller.record_evaluation(status == "PROMOTED")

                history.append({
                    "generation": gen,
                    "model_exponents": best_model.exponents.copy(),
                    "score": best_score,
                    "status": status,
                    "theta_promote": theta_promote
                })

                # Scale mutation step-size up (success)
                sigma = min(sigma * 1.1, 1.5)
                stagnation_counter = 0

                # Stop search early if model has been promoted
                if status == "PROMOTED":
                    break
            else:
                # Scale mutation step-size down (failure)
                sigma = max(sigma * 0.9, 0.05)
                stagnation_counter += 1

            # Restart mechanism if search stagnates
            if stagnation_counter >= self.restart_stagnation:
                # Reinitialize to a random power term
                random_exponent = rng.uniform(0.1, 5.0)
                best_model = MultiModalModel([FunctionalTerm("power", random_exponent)])
                best_model.fit(calib_x, calib_y, rng=rng)
                y_sim = internal_simulator(rng, best_model, ref_x)
                crps_val = compute_crps(y_sim, ref_y)
                penalty = get_complexity_penalty(best_model, len(ref_y), method=self.complexity_method)
                best_score = float(np.exp(-(crps_val + penalty)))

                sigma = 0.3
                stagnation_counter = 0

        return best_model, history
