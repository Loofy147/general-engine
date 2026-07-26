import numpy as np

def active_sampling_query(model, rng, candidate_points, n_samples=1):
    """
    Decide where to sample next using sequential greedy Bayesian Active Learning by Disagreement (BALD).
    BALD score: 0.5 * ln(1 + Var_epistemic(x) / sigma^2)
    Maximizes mutual information between parameters and outputs while ensuring a space-filling distribution.
    """
    remaining = np.array(candidate_points).copy()
    selected = []

    # Calculate the exclusion radius dynamically based on candidate space range
    if len(remaining) > 1:
        min_dist = (np.max(remaining) - np.min(remaining)) * 0.10
    else:
        min_dist = 0.5

    # Retrieve bootstrap samples from model
    if getattr(model, 'k_bootstrap', None) is not None:
        k_samples = model.k_bootstrap
    else:
        # Fallback to random normal samples if bootstrap is missing
        k_samples = rng.normal(model.k[:, None], np.abs(model.k[:, None]) * 0.1 + 1e-9, size=(len(model.k), 100))

    sigma = getattr(model, 'residual_scale', 0.10)
    sigma_sq = max(sigma ** 2, 1e-9)

    for _ in range(n_samples):
        if len(remaining) == 0:
            break

        # Point predictions for all parameter samples
        # model(remaining, k=k_samples) has shape (len(remaining), M)
        y_pred_samples = model(remaining, k=k_samples)

        # Epistemic variance (disagreement between parameter samples)
        var_epistemic = np.var(y_pred_samples, axis=1)

        # Compute BALD scores
        bald_scores = 0.5 * np.log(1.0 + var_epistemic / sigma_sq)

        # Pick the point with the highest BALD score
        best_idx = np.argmax(bald_scores)
        chosen_val = remaining[best_idx]
        selected.append(chosen_val)

        # Space-filling: remove points within the exclusion radius
        mask = np.abs(remaining - chosen_val) >= min_dist
        remaining = remaining[mask]

    return np.array(selected)
