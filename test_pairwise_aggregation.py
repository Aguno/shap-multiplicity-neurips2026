"""Small deterministic checks; no project results, training, or null simulation."""

import unittest

import numpy as np

from pairwise_aggregation import (
    pairwise_distances, pairwise_exceedance, validate_replicates,
)


class PairwiseAggregationTests(unittest.TestCase):
    def test_b_counts_events_before_averaging(self):
        distances = np.zeros((10, 3))
        distances[:4, 0] = 1.0
        distances[:, 1] = 0.6
        result = pairwise_exceedance(distances, 0.5)
        self.assertEqual(result["n_exceed"], 14)
        self.assertEqual(result["n_observations"], 30)
        self.assertAlmostEqual(result["percent"], 100 * 14 / 30)
        self.assertNotAlmostEqual(result["percent"],
                                  100 * np.mean(distances.mean(axis=0) > 0.5))

    def test_threshold_is_strict_and_zero_is_valid(self):
        result = pairwise_exceedance(np.full((10, 2), 0.5), 0.5)
        self.assertEqual(result["percent"], 0.0)
        self.assertEqual(result["n_instances"], 2)

    def test_rejects_partial_pairs_flat_arrays_and_nonfinite_values(self):
        for values in [np.zeros((9, 2)), np.zeros(20), np.zeros((10, 0)),
                       np.full((10, 2), np.nan), np.full((10, 2), np.inf)]:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                pairwise_exceedance(values, 0.5)
        with self.assertRaises(ValueError):
            pairwise_exceedance(np.zeros((10, 2)), np.nan)

    def test_rejects_missing_or_unequal_runs(self):
        values = [np.zeros((3, 2)) for _ in range(5)]
        for invalid in [values[:4], values[:4] + [None],
                        values[:4] + [np.zeros((2, 2))],
                        values[:4] + [np.full((3, 2), np.nan)]]:
            with self.assertRaises(ValueError):
                validate_replicates(invalid)

    def test_pair_axis_preserves_all_instances(self):
        values = [np.full((2, 1), i) for i in range(5)]
        distances = pairwise_distances(values, lambda a, b: np.abs(a[:, 0] - b[:, 0]))
        self.assertEqual(distances.shape, (10, 2))
        self.assertEqual(distances[:, 0].tolist(), [1, 2, 3, 4, 1, 2, 3, 1, 2, 1])
        self.assertEqual(pairwise_exceedance(distances, 2)["n_exceed"], 6)


if __name__ == "__main__":
    unittest.main()
