import numpy as np
import uuid
import time
from engine.core import MultiModalModel, hybrid_verisimilitude, r2_verisimilitude, coverage_verisimilitude, max_error_ratio, internal_simulator
from engine.sandbox import EvolutionarySandbox, mutate_model
from engine.curiosity import CuriosityController

# ================================================================
# Canonical Event Schema and Ingestion Plane
# ================================================================
class IngestionPlane:
    """
    Validates and normalizes all raw inputs into canonical event schema:
    {
        "event_id": str (UUID),
        "timestamp": float (epoch),
        "source": str,
        "event_type": str,
        "data": dict,
        "metadata": dict
    }
    """
    def normalize_and_validate(self, raw_input: dict) -> dict:
        required_fields = ["source", "event_type", "data"]
        for f in required_fields:
            if f not in raw_input:
                raise ValueError(f"Missing required field '{f}' in raw input.")

        canonical_event = {
            "event_id": raw_input.get("event_id", str(uuid.uuid4())),
            "timestamp": raw_input.get("timestamp", time.time()),
            "source": raw_input["source"],
            "event_type": raw_input["event_type"],
            "data": dict(raw_input["data"]),
            "metadata": dict(raw_input.get("metadata", {}))
        }
        return canonical_event


# ================================================================
# Policy Plane
# ================================================================
class RoutingPolicy:
    """
    A routing policy that decides whether to route an experiment request
    to 'cheap_unreliable_sensor' or 'expensive_precise_sensor'.
    Its functional parameter is 'threshold'.
    """
    def __init__(self, threshold=2.0):
        self.threshold = float(threshold)

    def route(self, x_val: float) -> str:
        # If x_val > threshold, route to expensive precise sensor
        # otherwise cheap unreliable sensor.
        if x_val > self.threshold:
            return "expensive_precise_sensor"
        return "cheap_unreliable_sensor"

    def mutate(self, rng) -> 'RoutingPolicy':
        # Mutate the threshold
        new_thresh = np.clip(self.threshold + rng.normal(0, 0.5), 0.1, 10.0)
        return RoutingPolicy(new_thresh)


class PolicyPlane:
    """
    Manages active policies in the substrate (e.g. routing policies, filters).
    Allows storing and retrieving policies that are subjected to the meta-loop.
    """
    def __init__(self):
        self.active_policies = {}

    def set_policy(self, name: str, policy):
        self.active_policies[name] = policy

    def get_policy(self, name: str):
        return self.active_policies.get(name)


# ================================================================
# Execution Plane
# ================================================================
class ExecutionPlane:
    """
    Executes simulated actions, supports rollbacks and digital twin replays.
    """
    def __init__(self, ground_truth_fn):
        self.ground_truth = ground_truth_fn
        self.rollback_stack = []

    def execute(self, action_event: dict, policy: RoutingPolicy, rng) -> dict:
        """
        Executes a measurement query according to the routing policy.
        """
        x_val = action_event["data"]["x"]
        sensor = policy.route(x_val)

        # Save state for rollback capability
        self.rollback_stack.append({
            "action_event": action_event,
            "policy_threshold": policy.threshold,
            "sensor": sensor
        })

        y_true = self.ground_truth(x_val)

        # Simulate sensor characteristics
        if sensor == "expensive_precise_sensor":
            # Very precise Gaussian noise
            noise = rng.normal(0, 0.02 * y_true + 1e-9)
            cost = 10.0
        else:
            # Cheap and noisy with occasional outliers (heavy-tailed)
            if rng.random() < 0.15: # 15% outlier rate
                noise = rng.standard_cauchy() * 0.5 * y_true
            else:
                noise = rng.normal(0, 0.15 * y_true + 1e-9)
            cost = 1.0

        y_measured = float(y_true + noise)

        result_event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "source": "ExecutionPlane",
            "event_type": "measurement_result",
            "data": {
                "x": x_val,
                "y": y_measured,
                "sensor_used": sensor,
                "cost": cost
            },
            "metadata": {
                "parent_event_id": action_event["event_id"]
            }
        }
        return result_event

    def rollback_last_action(self):
        """
        Rolls back the last execution action.
        """
        if self.rollback_stack:
            return self.rollback_stack.pop()
        return None

    def digital_twin_replay(self, rng, event_log: list, policy: RoutingPolicy) -> list:
        """
        Replays past logged measurement events using a new candidate policy.
        """
        simulated_results = []
        for event in event_log:
            if event["event_type"] == "measurement_request":
                x_val = event["data"]["x"]
                sensor = policy.route(x_val)
                y_true = self.ground_truth(x_val)
                if sensor == "expensive_precise_sensor":
                    noise = rng.normal(0, 0.02 * y_true + 1e-9)
                    cost = 10.0
                else:
                    if rng.random() < 0.15:
                        noise = rng.standard_cauchy() * 0.5 * y_true
                    else:
                        noise = rng.normal(0, 0.15 * y_true + 1e-9)
                    cost = 1.0
                simulated_results.append({
                    "x": x_val,
                    "y_simulated": y_true + noise,
                    "sensor": sensor,
                    "cost": cost
                })
        return simulated_results


# ================================================================
# Audit Plane
# ================================================================
class AuditPlane:
    """
    Maintains immutable event log, performance statistics, and drift detection.
    """
    def __init__(self):
        self.immutable_log = []

    def log_event(self, event: dict):
        # Enforce immutability via deep copy simulation
        frozen_event = {
            "event_id": event["event_id"],
            "timestamp": event["timestamp"],
            "source": event["source"],
            "event_type": event["event_type"],
            "data": dict(event["data"]),
            "metadata": dict(event["metadata"])
        }
        self.immutable_log.append(frozen_event)

    def detect_drift(self, metric_key: str, window_size=20, threshold=2.0) -> bool:
        """
        Detects drift in the mean of observed values in the event log.
        Compares the latest window_size elements to the preceding window_size elements.
        """
        values = []
        for e in self.immutable_log:
            if e["event_type"] == "measurement_result" and metric_key in e["data"]:
                values.append(e["data"][metric_key])

        if len(values) < 2 * window_size:
            return False # Not enough data to determine drift reliably

        preceding = values[-2*window_size : -window_size]
        current = values[-window_size:]

        mean_prec = np.mean(preceding)
        mean_curr = np.mean(current)
        std_prec = np.std(preceding) if np.std(preceding) > 1e-9 else 1.0

        z_score = np.abs(mean_curr - mean_prec) / std_prec
        return bool(z_score > threshold)

    def get_performance_report(self) -> dict:
        """
        Computes total cost, sensor usage counts, and average error.
        """
        costs = []
        sensors = {}
        for e in self.immutable_log:
            if e["event_type"] == "measurement_result":
                costs.append(e["data"]["cost"])
                s = e["data"]["sensor_used"]
                sensors[s] = sensors.get(s, 0) + 1

        return {
            "total_events": len(self.immutable_log),
            "total_cost": float(np.sum(costs)) if costs else 0.0,
            "average_cost": float(np.mean(costs)) if costs else 0.0,
            "sensor_distribution": sensors
        }


# ================================================================
# Knowledge Plane
# ================================================================
class KnowledgePlane:
    """
    Hosts the core scientific models, policy evaluations, and adaptive curiosity.
    """
    def __init__(self, initial_model: MultiModalModel, complexity_method="BIC"):
        self.model = initial_model
        self.complexity_method = complexity_method
        self.curiosity_controller = CuriosityController(window_size=10, initial_success_rate=0.5)
        self.sandbox = EvolutionarySandbox(max_generations=20, B=50, complexity_method=complexity_method)
        # Store policies and A/B test results
        self.policy_score_history = {}

        # Cooldown parameters
        self.is_cooldown_active = False
        self.fresh_samples_since_promotion = 0

    def update_scientific_model(self, rng, calib_x, calib_y, data_generator_fn):
        """
        Evolve our structural scientific model using the Evolutionary Sandbox.
        If cooldown is active, skip demotion / status re-evaluation.
        """
        if self.is_cooldown_active:
            if self.fresh_samples_since_promotion >= 3:
                # Fresh dataset has been collected, clear the cooldown
                self.is_cooldown_active = False
                self.fresh_samples_since_promotion = 0
            else:
                # Bypass search / demotion during cooldown
                return self.model, [{
                    "generation": 0,
                    "model_exponents": self.model.exponents.copy(),
                    "score": 1.0,
                    "status": "PROMOTED",
                    "theta_promote": 0.85,
                    "note": "cooldown active (skip re-evaluation)"
                }]

        self.model, history = self.sandbox.run(
            rng, calib_x, calib_y, self.model, self.curiosity_controller, data_generator_fn
        )

        # If the search results in a PROMOTED status, activate the cooldown
        if history and history[-1]["status"] == "PROMOTED":
            self.is_cooldown_active = True
            self.fresh_samples_since_promotion = 0

        return self.model, history

    def evaluate_policy_performance(self, rng, execution_plane, audit_log, policy: RoutingPolicy) -> float:
        """
        Evaluates a candidate policy via Digital Twin Replay over logged events,
        scoring it using a verisimilitude-like metric of output accuracy vs cost.
        """
        replayed = execution_plane.digital_twin_replay(rng, audit_log, policy)
        if not replayed:
            return 0.0

        # We want to maximize precision while minimizing cost.
        # Score = Average Accuracy (1 - relative error) - lambda * Normalized Cost
        costs = [item["cost"] for item in replayed]
        xs = np.array([item["x"] for item in replayed])
        ys_sim = np.array([item["y_simulated"] for item in replayed])

        # Evaluate the scientific model accuracy on the replayed data
        y_model_pred = self.model(xs)
        rel_errors = np.abs(ys_sim - y_model_pred) / (np.abs(ys_sim) + 1e-9)
        avg_precision = max(0.0, 1.0 - float(np.mean(rel_errors)))

        # Normalized cost: cheap sensor cost=1.0, expensive sensor cost=10.0
        # If mean cost is 1.0, cost_penalty is 0. If mean cost is 10.0, cost_penalty is 0.5.
        avg_cost = np.mean(costs)
        cost_penalty = 0.05 * (avg_cost - 1.0) # Penalty up to 0.45

        policy_score = float(np.clip(avg_precision - cost_penalty, 0.0, 1.0))
        return policy_score


# ================================================================
# Meta-Loop (Orchestrator)
# ================================================================
class MetaLoopSubstrate:
    """
    Integrates all five planes into a unified self-experimenting substrate.
    """
    def __init__(self, ground_truth_fn, initial_model: MultiModalModel, complexity_method="BIC"):
        self.ingestion = IngestionPlane()
        self.policy_plane = PolicyPlane() # Holds active policies
        self.active_policy = RoutingPolicy(threshold=2.0) # Default routing policy
        self.policy_plane.set_policy("routing", self.active_policy)
        self.execution = ExecutionPlane(ground_truth_fn)
        self.audit = AuditPlane()
        self.knowledge = KnowledgePlane(initial_model, complexity_method)

    def process_measurement_request(self, rng, raw_request: dict) -> dict:
        """
        Runs the standard pipeline: Ingest -> Apply Policy -> Execute -> Audit.
        """
        # 1. Ingestion Plane
        canonical_req = self.ingestion.normalize_and_validate(raw_request)
        self.audit.log_event(canonical_req)

        # 2. Execution Plane (uses the Policy Plane's active policy)
        result_event = self.execution.execute(canonical_req, self.active_policy, rng)

        # Track fresh samples collected since promotion for cooldown
        if self.knowledge.is_cooldown_active:
            self.knowledge.fresh_samples_since_promotion += 1

        # 3. Audit Plane
        self.audit.log_event(result_event)
        return result_event

    def self_experiment_and_optimize_policies(self, rng, n_mutations=10):
        """
        The Meta-Loop: treating routing policies as models, A/B testing,
        evaluating via Digital Twin replay in the Knowledge Plane, and mutating/promoting them.
        """
        current_score = self.knowledge.evaluate_policy_performance(
            rng, self.execution, self.audit.immutable_log, self.active_policy
        )

        best_policy = self.active_policy
        best_score = current_score

        optimization_log = []

        for m in range(n_mutations):
            candidate_policy = best_policy.mutate(rng)
            candidate_score = self.knowledge.evaluate_policy_performance(
                rng, self.execution, self.audit.immutable_log, candidate_policy
            )

            # Accept if confidently better (A/B testing outcome)
            if candidate_score > best_score:
                best_policy = candidate_policy
                best_score = candidate_score
                optimization_log.append({
                    "step": m + 1,
                    "old_threshold": best_policy.threshold,
                    "new_threshold": candidate_policy.threshold,
                    "new_score": candidate_score,
                    "status": "IMPROVED"
                })

        self.active_policy = best_policy
        self.policy_plane.set_policy("routing", self.active_policy)
        return best_score, optimization_log


# ================================================================
# Decoupled Oracle Provider and Conformal Calibration
# ================================================================
class OracleProvider:
    def __init__(self, ground_truth_fn):
        self.ground_truth_fn = ground_truth_fn

    def query_oracle(self, x: float) -> float:
        """
        Returns the trusted, high-precision ground truth reference for x.
        """
        return float(self.ground_truth_fn(x))


class ConformalCalibrator:
    """
    Implements Split Conformal Prediction to calibrate the engine's predictive intervals
    against trusted oracle calibration points.
    """
    def __init__(self, target_coverage=0.90):
        self.target_coverage = target_coverage
        self.calibration_points = []
        self.q_scale = 1.0  # Conformal scaling multiplier

    def register_calibration_point(self, x: float, y_true: float):
        """
        Register a new ground-truth calibration checkpoint from the oracle.
        """
        self.calibration_points.append((x, y_true))

    def update_calibration(self, model, rng):
        """
        Computes non-conformity scores on calibration set and updates q_scale.
        non_conformity s_i = |y_i - y_pred_i| / std_pred_i
        q_scale is the (1 - alpha)*(1 + 1/n) quantile of s_i.
        """
        n = len(self.calibration_points)
        if n < 3:
            self.q_scale = 1.0  # Not enough points, keep unscaled
            return

        xs = np.array([pt[0] for pt in self.calibration_points])
        ys = np.array([pt[1] for pt in self.calibration_points])

        # Get predictions and simulator standard deviations
        y_pred = model(xs)
        y_sims = internal_simulator(rng, model, xs, n_iter=200)
        y_stds = np.std(y_sims, axis=1)
        y_stds = np.clip(y_stds, 1e-4, None) # Avoid divide by zero

        # Compute non-conformity scores
        non_conformity_scores = np.abs(ys - y_pred) / y_stds

        # Quantile index: ceil((n + 1) * target_coverage) / n
        pct = 100.0 * (self.target_coverage * (n + 1) / n)
        pct = np.clip(pct, 0.0, 100.0)
        self.q_scale = float(np.percentile(non_conformity_scores, pct))

    def calibrate_interval(self, y_pred, y_sims):
        """
        Applies the conformal scaling factor to the predictive simulations.
        """
        errors = y_sims - y_pred[:, None]
        return y_pred[:, None] + self.q_scale * errors
