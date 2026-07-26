import unittest
import numpy as np
from engine.core import (
    MultiModalModel, internal_simulator, r2_verisimilitude,
    coverage_verisimilitude, max_error_ratio, compute_aic, compute_bic,
    get_complexity_penalty, hybrid_verisimilitude, classify_model
)
from engine.curiosity import CuriosityController
from engine.active_sampling import active_sampling_query
from engine.sandbox import EvolutionarySandbox, mutate_model, estimate_prob_better
from engine.substrate import IngestionPlane, RoutingPolicy, ExecutionPlane, AuditPlane, KnowledgePlane, MetaLoopSubstrate
from engine.pilot import EnzymeKineticsPilot, enzyme_ground_truth, query_enzyme_data


class TestEngineCore(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_model_fitting_and_predictions(self):
        # Generate clean synthetic data: y = 2.5 * x^2
        x = np.linspace(0.5, 5.0, 20)
        y = 2.5 * x**2

        model = MultiModalModel([2.0])
        model.fit(x, y)

        self.assertAlmostEqual(model.k[0], 2.5, places=4)
        self.assertIsNotNone(model.cov_matrix)
        self.assertEqual(model.cov_matrix.shape, (1, 1))

        # Test predictions
        y_pred = model(x)
        self.assertEqual(len(y_pred), len(x))
        np.testing.assert_allclose(y_pred, y, rtol=1e-3)

    def test_internal_simulator(self):
        x = np.linspace(1.0, 5.0, 10)
        model = MultiModalModel([2.0])
        model.fit(x, 2.5 * x**2)

        y_sim = internal_simulator(self.rng, model, x, n_iter=200)
        self.assertEqual(y_sim.shape, (10, 200))

    def test_metrics_and_verisimilitude(self):
        x = np.linspace(1.0, 5.0, 10)
        y_true = 2.5 * x**2
        model = MultiModalModel([2.0])
        model.fit(x, y_true)

        r2 = r2_verisimilitude(model, x, y_true)
        self.assertGreaterEqual(r2, 0.95)

        y_sim = internal_simulator(self.rng, model, x, n_iter=1000)
        cov = coverage_verisimilitude(y_sim, y_true)
        self.assertGreaterEqual(cov, 0.80) # 90% predictive interval coverage should be high

        mer = max_error_ratio(model, x, y_true)
        self.assertLess(mer, 0.10)

        # Test AIC and BIC
        aic = compute_aic(model, x, y_true)
        bic = compute_bic(model, x, y_true)
        self.assertIsNotNone(aic)
        self.assertIsNotNone(bic)

        # Test complexity penalties
        p_bic = get_complexity_penalty(model, len(x), "BIC")
        p_aic = get_complexity_penalty(model, len(x), "AIC")
        self.assertGreater(p_bic, 0)
        self.assertGreater(p_aic, 0)

        # Test hybrid score with penalties
        score = hybrid_verisimilitude(r2, cov, mer, lam=0.25, complexity_penalty=p_bic)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestCuriosityController(unittest.TestCase):
    def test_adaptive_threshold(self):
        controller = CuriosityController(window_size=5, initial_success_rate=1.0)
        self.assertAlmostEqual(controller.get_threshold(), 0.85 + 0.10 * np.tanh(2.5), places=3)

        # Fill window with False (rejections)
        for _ in range(5):
            controller.record_evaluation(False)

        self.assertAlmostEqual(controller.success_rate, 0.0)
        # s=0.0 -> tanh(-2.5) -> threshold should lower to near 0.75
        self.assertLess(controller.get_threshold(), 0.78)


class TestActiveSampling(unittest.TestCase):
    def test_space_filling(self):
        rng = np.random.default_rng(42)
        model = MultiModalModel([2.0])
        model.k = np.array([2.5])
        model.cov_matrix = np.array([[0.01]])

        candidate_points = np.linspace(1.0, 10.0, 50)
        # Select 3 points
        selected = active_sampling_query(model, rng, candidate_points, n_samples=3)
        self.assertEqual(len(selected), 3)

        # Check exclusion: minimum distance between selected points should be at least 10% of 9.0 range (= 0.9)
        dists = [np.abs(selected[0] - selected[1]), np.abs(selected[1] - selected[2]), np.abs(selected[0] - selected[2])]
        for d in dists:
            self.assertGreaterEqual(d, 0.90)


class TestEvolutionarySandbox(unittest.TestCase):
    def test_mutations(self):
        rng = np.random.default_rng(99)
        model = MultiModalModel([1.0, 2.0])
        mutated = mutate_model(model, rng)
        self.assertIsNotNone(mutated)
        self.assertTrue(1 <= len(mutated.exponents) <= 3)

    def test_sandbox_run(self):
        rng = np.random.default_rng(42)
        controller = CuriosityController(window_size=5, initial_success_rate=0.5)
        sandbox = EvolutionarySandbox(max_generations=5, B=10, complexity_method="BIC")

        cal_x = np.linspace(1.0, 5.0, 10)
        cal_y = 3.0 * cal_x**2

        def mock_data_gen(r, size):
            x = r.uniform(1.0, 5.0, size=size)
            return x, 3.0 * x**2

        initial_model = MultiModalModel([1.0])
        best_model, history = sandbox.run(rng, cal_x, cal_y, initial_model, controller, mock_data_gen)

        self.assertIsNotNone(best_model)
        self.assertGreater(len(history), 0)


class TestSubstrateAndMetaLoop(unittest.TestCase):
    def test_five_planes_and_meta_loop(self):
        rng = np.random.default_rng(100)

        # Instantiate pilot substrate using Enzyme Kinetics MM curve
        pilot = EnzymeKineticsPilot(complexity_method="BIC")

        # 1. Ingestion Plane raw validation
        raw_req = {"source": "manual", "event_type": "measurement_request", "data": {"x": 2.5}}
        canonical_event = pilot.substrate.ingestion.normalize_and_validate(raw_req)
        self.assertEqual(canonical_event["source"], "manual")

        # 2. Execution Plane
        res = pilot.substrate.execution.execute(canonical_event, pilot.substrate.active_policy, rng)
        self.assertEqual(res["event_type"], "measurement_result")
        self.assertIn("sensor_used", res["data"])

        # Test execution rollback
        self.assertGreater(len(pilot.substrate.execution.rollback_stack), 0)
        rolled_back = pilot.substrate.execution.rollback_last_action()
        self.assertIsNotNone(rolled_back)
        self.assertEqual(rolled_back["sensor"], res["data"]["sensor_used"])

        # 3. Audit Plane logging and drift detection
        audit = pilot.substrate.audit
        # Log a bunch of events
        for val in range(10):
            audit.log_event({
                "event_id": f"evt-{val}",
                "timestamp": 12345.0,
                "source": "sensor",
                "event_type": "measurement_result",
                "data": {"y": float(val), "cost": 1.0},
                "metadata": {}
            })
        # Add drifted values
        for val in range(10):
            audit.log_event({
                "event_id": f"evt-drift-{val}",
                "timestamp": 12345.0,
                "source": "sensor",
                "event_type": "measurement_result",
                "data": {"y": float(val + 20.0), "cost": 1.0},
                "metadata": {}
            })
        has_drift = audit.detect_drift("y", window_size=5, threshold=1.5)
        self.assertTrue(has_drift)

        # 4. Meta-Loop policy optimization
        # Collect some workflow requests first so we have re-playable logs
        for i in range(5):
            pilot.substrate.process_measurement_request(rng, {
                "source": "test_suite",
                "event_type": "measurement_request",
                "data": {"x": 3.0}
            })

        initial_threshold = pilot.substrate.active_policy.threshold
        best_score, opt_log = pilot.substrate.self_experiment_and_optimize_policies(rng, n_mutations=5)
        self.assertIsNotNone(best_score)


if __name__ == "__main__":
    unittest.main()
