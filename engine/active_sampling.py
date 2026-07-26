import numpy as np
from engine.core import internal_simulator

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
    if len(remaining) > 1:
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
