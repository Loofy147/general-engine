import numpy as np

class CuriosityController:
    """
    Implements the Curiosity Threshold (Meta-Layer) described in Section 2.7.
    Adapts the promotion threshold based on the promotion success rate (s)
    within a sliding window.
    """
    def __init__(self, window_size=10, initial_success_rate=0.5):
        self.window_size = window_size
        # Seed the window with initial_success_rate values to avoid cold-start issues
        self.window = [True if i < int(window_size * initial_success_rate) else False for i in range(window_size)]

    def record_evaluation(self, is_promoted: bool):
        """
        Records the outcome of a model promotion attempt.
        True for PROMOTED, False otherwise.
        """
        self.window.append(is_promoted)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    @property
    def success_rate(self) -> float:
        """
        Current success rate (s) in the sliding window.
        """
        if not self.window:
            return 0.5
        return float(np.mean(self.window))

    def get_threshold(self) -> float:
        """
        Computes the adaptive promotion threshold:
        theta_promote(s) = 0.85 + 0.10 * tanh((s - 0.5) / 0.2)
        """
        s = self.success_rate
        threshold = 0.85 + 0.10 * np.tanh((s - 0.5) / 0.2)
        # Ensure it stays within reasonable bounds [0.75, 0.95]
        return float(np.clip(threshold, 0.75, 0.95))
