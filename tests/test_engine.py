import unittest
import numpy as np
from engine.core import (
    MultiModalModel, FunctionalTerm, MultivariateFunctionalTerm, internal_simulator, r2_verisimilitude,
    coverage_verisimilitude, max_error_ratio, compute_aic, compute_bic,
    get_complexity_penalty, hybrid_verisimilitude, classify_model,
    compute_crps, check_bayesian_coverage, check_pit_calibration
)
from engine.curiosity import CuriosityController
from engine.active_sampling import active_sampling_query
from engine.sandbox import EvolutionarySandbox, mutate_model, estimate_prob_better
from engine.substrate import (
    IngestionPlane, RoutingPolicy, ExecutionPlane, AuditPlane, KnowledgePlane,
    MetaLoopSubstrate, OracleProvider, ConformalCalibrator, KaggleDataSource
)
from engine.pilot import EnzymeKineticsPilot, enzyme_ground_truth, query_enzyme_data


class TestEngineCore(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_model_fitting_and_predictions(self):
        # Generate clean synthetic data: y = 2.5 * x^2
        x = np.linspace(0.5, 5.0, 20)
        y = 2.5 * x**2

        # Model with single power term (p=2)
        model = MultiModalModel([2.0])
        model.fit(x, y, rng=self.rng)

        self.assertAlmostEqual(model.k[0], 2.5, places=2)
        self.assertIsNotNone(model.k_bootstrap)

        # Test predictions
        y_pred = model(x)
        self.assertEqual(len(y_pred), len(x))
        np.testing.assert_allclose(y_pred, y, rtol=1e-2)

    def test_functional_terms_exp_log(self):
        # Test exponential term
        term_exp = FunctionalTerm("exp", 1.5)
        self.assertAlmostEqual(term_exp.evaluate(0.0)[0], 1.0)
        self.assertAlmostEqual(term_exp.evaluate(1.0)[0], np.exp(-1.5))

        # Test log term
        term_log = FunctionalTerm("log", 2.0)
        self.assertAlmostEqual(term_log.evaluate(0.0)[0], 0.0)
        self.assertAlmostEqual(term_log.evaluate(1.0)[0], np.log(3.0))

        # MultiModalModel with mixed terms
        model = MultiModalModel([
            FunctionalTerm("power", 1.0),
            FunctionalTerm("exp", 0.5),
            FunctionalTerm("log", 1.0)
        ])
        x = np.linspace(0.1, 2.0, 10)
        design = model.get_design_matrix(x)
        self.assertEqual(design.shape, (10, 3))

    def test_internal_simulator(self):
        x = np.linspace(1.0, 5.0, 10)
        model = MultiModalModel([2.0])
        model.fit(x, 2.5 * x**2, rng=self.rng)

        y_sim = internal_simulator(self.rng, model, x, n_iter=200)
        self.assertEqual(y_sim.shape, (10, 200))

    def test_crps_scoring(self):
        # Deterministic check
        y_samples = np.array([[1.0, 5.0]])
        y_true = np.array([3.0])
        crps = compute_crps(y_samples, y_true)
        # E|X - y| = 0.5 * (|1 - 3| + |5 - 3|) = 2
        # E|X - X'| = 0.5 * (|1 - 1| + |1 - 5| + |5 - 1| + |5 - 5|) = 2
        # CRPS = E|X - y| - 0.5 * E|X - X'| = 2 - 0.5 * 2 = 1.0
        self.assertAlmostEqual(crps, 1.0)

    def test_bayesian_coverage_and_pit(self):
        ref_y = self.rng.normal(0, 1, size=30)
        y_sim = self.rng.normal(0, 1, size=(30, 1000))

        success, prob, emp = check_bayesian_coverage(y_sim, ref_y, target_coverage=0.80)
        self.assertTrue(success)
        self.assertGreater(prob, 0.50)

        passes_ks, p_val = check_pit_calibration(y_sim, ref_y)
        self.assertTrue(passes_ks)
        self.assertGreater(p_val, 0.05)


class TestCuriosityController(unittest.TestCase):
    def test_thompson_sampling_curiosity(self):
        rng = np.random.default_rng(12345)
        controller = CuriosityController(window_size=5, initial_success_rate=0.8, rng=rng)

        # Pull threshold several times
        thresholds = [controller.get_threshold() for _ in range(10)]
        for t in thresholds:
            self.assertIn(t, [0.75, 0.80, 0.85, 0.90, 0.95])

        # Record rejections
        for _ in range(20):
            controller.record_evaluation(False)

        # Success rate should decrease
        self.assertLess(controller.success_rate, 0.5)


class TestActiveSampling(unittest.TestCase):
    def test_bald_active_sampling(self):
        rng = np.random.default_rng(42)
        model = MultiModalModel([2.0])
        model.k = np.array([2.5])
        model.k_bootstrap = np.array([[2.4, 2.5, 2.6]])
        model.residual_scale = 0.10

        candidate_points = np.linspace(1.0, 10.0, 50)
        selected = active_sampling_query(model, rng, candidate_points, n_samples=3)
        self.assertEqual(len(selected), 3)

        # Check exclusion: minimum distance between selected points should be at least 10% of 9.0 range (= 0.9)
        dists = [np.abs(selected[0] - selected[1]), np.abs(selected[1] - selected[2]), np.abs(selected[0] - selected[2])]
        for d in dists:
            self.assertGreaterEqual(d, 0.90)


class TestConformalCalibration(unittest.TestCase):
    def test_conformal_calibrator_and_oracle(self):
        rng = np.random.default_rng(42)
        oracle = OracleProvider(lambda x: 3.0 * x**2)
        self.assertEqual(oracle.query_oracle(2.0), 12.0)

        calibrator = ConformalCalibrator(target_coverage=0.90)
        for x_val in [1.0, 2.0, 3.0, 4.0]:
            calibrator.register_calibration_point(x_val, oracle.query_oracle(x_val))

        model = MultiModalModel([2.0])
        model.k = np.array([2.9])
        model.k_bootstrap = np.array([[2.8, 2.9, 3.0]])
        model.residual_scale = 0.10

        calibrator.update_calibration(model, rng)
        self.assertGreater(calibrator.q_scale, 0.0)

        y_pred = np.array([12.0])
        y_sims = np.array([[11.8, 12.0, 12.2]])
        calibrated = calibrator.calibrate_interval(y_pred, y_sims)
        self.assertEqual(calibrated.shape, (1, 3))


class TestEvolutionarySandbox(unittest.TestCase):
    def test_mutations(self):
        rng = np.random.default_rng(99)
        model = MultiModalModel([1.0, 2.0])
        mutated = mutate_model(model, rng)
        self.assertIsNotNone(mutated)
        self.assertTrue(1 <= len(mutated.terms) <= 3)

    def test_sandbox_run(self):
        rng = np.random.default_rng(42)
        controller = CuriosityController(window_size=5, initial_success_rate=0.5, rng=rng)
        sandbox = EvolutionarySandbox(max_generations=3, B=5, complexity_method="BIC")

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

        # Test cooldown period
        pilot.substrate.knowledge.is_cooldown_active = True
        pilot.substrate.knowledge.fresh_samples_since_promotion = 0
        # Call update_scientific_model, it should return early due to cooldown
        _, hist = pilot.substrate.knowledge.update_scientific_model(
            rng, np.array([1, 2]), np.array([3, 4]), lambda r, s: (np.array([1, 2]), np.array([3, 4]))
        )
        self.assertEqual(hist[0].get("note"), "cooldown active (skip re-evaluation)")

    def test_multivariate_functional_term(self):
        term = MultivariateFunctionalTerm([[0, "power", 2.0], [1, "log", 1.0]])
        X = np.array([[2.0, 1.0], [3.0, 2.0]])
        expected = np.array([4.0 * np.log(2.0), 9.0 * np.log(3.0)])
        np.testing.assert_allclose(term.evaluate(X), expected, rtol=1e-5)

    def test_multivariate_mutation(self):
        rng = np.random.default_rng(42)
        initial_model = MultiModalModel([
            MultivariateFunctionalTerm([[0, "power", 1.0]])
        ])
        mutated = mutate_model(initial_model, rng, num_features=8)
        self.assertIsNotNone(mutated)
        self.assertGreater(len(mutated.terms), 0)

    def test_kaggle_data_source(self):
        ds = KaggleDataSource("data/concrete_data.csv", "concrete_compressive_strength", ["cement", "water"])
        X, y = ds.load_data()
        self.assertEqual(X.shape[1], 2)
        self.assertEqual(len(y), len(X))


if __name__ == "__main__":
    unittest.main()
