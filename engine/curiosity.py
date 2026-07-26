import numpy as np

class CuriosityController:
    """
    Implements the Curiosity Threshold (Meta-Layer).
    Formalized as a multi-armed bandit using Thompson Sampling to choose
    the optimal promotion threshold, avoiding ad-hoc tanh functions,
    and adapting to non-stationary environments using a forgetting factor.
    """
    def __init__(self, window_size=10, initial_success_rate=0.5, rng=None):
        self.window_size = window_size
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Candidate threshold arms
        self.arms = [0.75, 0.80, 0.85, 0.90, 0.95]

        # Initialize Beta priors for each arm
        # To seed with initial_success_rate, we can set alpha and beta accordingly
        self.alphas = np.ones(len(self.arms)) * (initial_success_rate * 10.0)
        self.betas = np.ones(len(self.arms)) * ((1.0 - initial_success_rate) * 10.0)

        # Make sure alphas and betas are at least 1.0
        self.alphas = np.clip(self.alphas, 1.0, None)
        self.betas = np.clip(self.betas, 1.0, None)

        self.last_selected_arm_idx = 2  # Start with 0.85
        self.window = []

    def record_evaluation(self, is_promoted: bool):
        """
        Records the outcome of a model promotion attempt.
        Applies a forgetting factor to adapt to non-stationary environments.
        """
        self.window.append(is_promoted)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        # Apply exponential forgetting factor (e.g., 0.95) to historic beliefs
        decay = 0.95
        self.alphas = 1.0 + (self.alphas - 1.0) * decay
        self.betas = 1.0 + (self.betas - 1.0) * decay

        # Thompson Sampling update for the selected arm
        idx = self.last_selected_arm_idx
        if is_promoted:
            self.alphas[idx] += 1.0
        else:
            self.betas[idx] += 1.0

    @property
    def success_rate(self) -> float:
        """
        Current success rate in the sliding window for backward compatibility.
        """
        if not self.window:
            # Fallback to the mean success rate of the active arm
            idx = self.last_selected_arm_idx
            return float(self.alphas[idx] / (self.alphas[idx] + self.betas[idx]))
        return float(np.mean(self.window))

    def get_threshold(self) -> float:
        """
        Thompson Sampling action selection: samples from the Beta posterior
        for each threshold arm and returns the one with the highest success belief.
        """
        # Draw a sample from each arm's posterior
        samples = [self.rng.beta(a, b) for a, b in zip(self.alphas, self.betas)]

        # Select the arm with the highest sample (highest probability of success)
        self.last_selected_arm_idx = int(np.argmax(samples))
        return float(self.arms[self.last_selected_arm_idx])
