import unittest

import numpy as np

import torch

from fusion import build_router_features, degree_router, equal_average, learned_router, oracle_best_expert
from run_experiments import split_validation_masks


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = [
            np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]]),
            np.array([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]),
            np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]]),
        ]

    def test_equal_average_is_normalized(self):
        result = equal_average(self.probabilities)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_degree_router_is_normalized(self):
        result = degree_router(self.probabilities, np.array([1, 2, 5, 10]))
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)
        self.assertGreater(result.weights[0, 0], result.weights[-1, 0])

    def test_learned_router_is_normalized(self):
        probabilities = [np.tile(values, (3, 1)) for values in self.probabilities]
        labels = np.array([0, 0, 1, 1] * 3)
        train = np.zeros(12, dtype=bool)
        validation = np.zeros(12, dtype=bool)
        train[:6] = True
        validation[6:10] = True
        result = learned_router(
            probabilities, np.arange(12), labels, train, validation,
            seed=42, device=torch.device("cpu"), epochs=5, patience=2,
        )
        self.assertEqual(result.weights.shape, (12, 3))
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_oracle_is_correct_when_any_expert_is_correct(self):
        labels = np.array([0, 1, 1, 1])
        result = oracle_best_expert(self.probabilities, labels)
        expected = np.any(
            np.stack([values.argmax(axis=1) for values in self.probabilities], axis=1)
            == labels[:, None],
            axis=1,
        )
        np.testing.assert_array_equal(result.any_correct, expected)
        np.testing.assert_array_equal(
            result.probabilities.argmax(axis=1) == labels, expected
        )

    def test_router_feature_ablations_have_expected_widths(self):
        mask = np.ones(4, dtype=bool)
        degree = np.arange(4)
        self.assertEqual(build_router_features(self.probabilities, degree, mask, "degree").shape[1], 1)
        self.assertEqual(build_router_features(self.probabilities, degree, mask, "confidence").shape[1], 6)
        self.assertEqual(build_router_features(self.probabilities, degree, mask, "all").shape[1], 10)

    def test_validation_subsets_are_disjoint_and_complete(self):
        labels = np.repeat(np.arange(2), 10)
        official = torch.ones(20, dtype=torch.bool)
        base, router_train, router_stop = split_validation_masks(labels, official, 42)
        base = base.numpy()
        self.assertFalse(np.any(base & router_train))
        self.assertFalse(np.any(base & router_stop))
        self.assertFalse(np.any(router_train & router_stop))
        np.testing.assert_array_equal(base | router_train | router_stop, official.numpy())


if __name__ == "__main__":
    unittest.main()
