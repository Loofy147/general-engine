import numpy as np
from engine.core import (
    MultiModalModel, FunctionalTerm, MultivariateFunctionalTerm,
    internal_simulator, classify_model, get_complexity_penalty, compute_crps
)

def mutate_model(model, rng, p_step=0.3, num_features=1):
    """
    Randomly add, remove, or mutate a term (power, exp, log) in the model,
    with full support for both univariate and multivariate terms.
    """
    new_terms = [t.copy() for t in model.terms]

    # Helper to generate a random spec
    def random_spec():
        feat_idx = int(rng.integers(0, num_features)) if num_features > 1 else 0
        t_type = rng.choice(["power", "exp", "log"])
        p_val = float(rng.uniform(0.1, 5.0))
        return [feat_idx, t_type, p_val]

    # Helper to generate a random term
    def random_term():
        if num_features > 1:
            # 70% chance of a multivariate term, 30% univariate (on a random feature)
            if rng.random() < 0.7:
                return MultivariateFunctionalTerm([random_spec()])
            else:
                feat_idx = int(rng.integers(0, num_features))
                t_type = rng.choice(["power", "exp", "log"])
                p_val = float(rng.uniform(0.1, 5.0))
                return MultivariateFunctionalTerm([[feat_idx, t_type, p_val]])
        else:
            t_type = rng.choice(["power", "exp", "log"])
            p_val = float(rng.uniform(0.1, 5.0))
            return FunctionalTerm(t_type, p_val)

    # 1. Add a term
    if rng.random() < 0.2 and len(new_terms) < 4:
        new_terms.append(random_term())
    # 2. Remove a term
    elif rng.random() < 0.1 and len(new_terms) > 1:
        idx = rng.integers(0, len(new_terms))
        new_terms.pop(idx)
    # 3. Mutate an existing term
    else:
        idx = rng.integers(0, len(new_terms))
        term = new_terms[idx]

        if isinstance(term, MultivariateFunctionalTerm):
            r = rng.random()
            if r < 0.15 and len(term.specs) < 3:
                term.specs.append(random_spec())
            elif r < 0.25 and len(term.specs) > 1:
                spec_idx = rng.integers(0, len(term.specs))
                term.specs.pop(spec_idx)
            else:
                spec_idx = rng.integers(0, len(term.specs))
                spec = term.specs[spec_idx]
                r_spec = rng.random()
                if r_spec < 0.6:
                    spec[2] = float(np.clip(spec[2] + rng.normal(0, p_step), 0.1, 5.0))
                elif r_spec < 0.8:
                    spec[1] = rng.choice(["power", "exp", "log"])
                    spec[2] = float(rng.uniform(0.1, 5.0))
                else:
                    if num_features > 1:
                        spec[0] = int(rng.integers(0, num_features))
        else:
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
        calib_x_arr = np.atleast_1d(calib_x)
        num_features = calib_x_arr.shape[1] if calib_x_arr.ndim > 1 else 1

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
            candidate = mutate_model(best_model, rng, p_step=sigma, num_features=num_features)
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
                if num_features > 1:
                    feat_idx = int(rng.integers(0, num_features))
                    random_exponent = rng.uniform(0.1, 5.0)
                    best_model = MultiModalModel([MultivariateFunctionalTerm([[feat_idx, "power", random_exponent]])])
                else:
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
