"""Focused RBO checks using synthetic rankings; no saved SHAP or model execution."""

import itertools
import random
import unittest
from unittest.mock import patch

import numpy as np

import baseline
import plot_violin
from pairwise_aggregation import pairwise_exceedance


def values_for_rankings(rankings):
    """Construct distinct positive magnitudes with the specified descending order."""
    rankings = np.asarray(rankings)
    values = np.empty(rankings.shape, dtype=float)
    for row, ranking in enumerate(rankings):
        values[row, ranking] = np.arange(len(ranking), 0, -1)
    return values


def truncated_distance(rank_a, rank_b, p):
    """Previous finite-sum definition, retained only to check the known shift."""
    return 1.0 - sum(
        (1.0 - p) * p ** (depth - 1)
        * len(set(rank_a[:depth]) & set(rank_b[:depth])) / depth
        for depth in range(1, len(rank_a) + 1)
    )


class RBOTests(unittest.TestCase):
    def test_identical_rankings_have_exact_zero_distance(self):
        for d in [1, 3, 8, 10, 16, 278]:
            with self.subTest(d=d):
                values = np.arange(d, dtype=float)[None, :]
                p = 1 - 1 / d
                np.testing.assert_array_equal(
                    plot_violin.compute_rbo_distance(values, values), [0.0])
                np.testing.assert_array_equal(
                    baseline.compute_rbo_distance(values, values, r=d, p=p), [0.0])
                self.assertEqual(
                    baseline.rbo_similarity_at_r(list(range(d)), list(range(d)), r=d, p=p),
                    1.0)

    def test_known_three_feature_swap(self):
        # Prefix overlaps are [0, 1, 1]; p=.5 gives similarity .5 including tail.
        a, b = values_for_rankings([[0, 1, 2], [1, 0, 2]])
        np.testing.assert_allclose(
            plot_violin.compute_rbo_similarity(a[None], b[None], p=.5), [.5])
        self.assertAlmostEqual(
            baseline.rbo_similarity_at_r([0, 1, 2], [1, 0, 2], r=3, p=.5), .5)

    def test_partial_depth_uses_last_overlap_in_tail(self):
        # At depth 2, A_1=1 and A_2=.5: .5 + .25*.5 + .25*.5 = .75.
        rank_a, rank_b = [0, 1, 2, 3], [0, 2, 3, 1]
        a, b = values_for_rankings([rank_a, rank_b])
        self.assertAlmostEqual(
            baseline.rbo_similarity_at_r(rank_a, rank_b, r=2, p=.5), .75)
        np.testing.assert_allclose(
            baseline.compute_rbo_similarity(a[None], b[None], r=2, p=.5), [.75])
        np.testing.assert_allclose(
            plot_violin.compute_rbo_similarity(a[None], b[None], depth=2, p=.5), [.75])

    def test_plotting_and_null_helpers_agree_for_all_small_rankings(self):
        rankings = list(itertools.permutations(range(4)))
        a = np.repeat(values_for_rankings([rankings[0]]), len(rankings), axis=0)
        b = values_for_rankings(rankings)
        for depth in [1, 2, 4, 100]:
            with self.subTest(depth=depth):
                expected = np.array([
                    baseline.rbo_similarity_at_r(rankings[0], ranking, r=depth, p=.75)
                    for ranking in rankings
                ])
                np.testing.assert_allclose(
                    baseline.compute_rbo_similarity(a, b, r=depth, p=.75), expected)
                np.testing.assert_allclose(
                    plot_violin.compute_rbo_similarity(a, b, depth=depth, p=.75), expected)

    def test_zero_dispersion_null_has_zero_distance(self):
        result = baseline.compute_rbo_baseline_simplified(
            d=16, r=16, p=15/16, q=0.0, n_trials=5, return_std=True)
        self.assertEqual(result["mean"], 0.0)
        self.assertEqual(result["E_sim"], 1.0)
        self.assertEqual(result["std"], 0.0)

    def test_full_ranking_shift_preserves_b_threshold_decisions(self):
        d, p, q, seed, n_trials = 4, .75, .4, 19, 64
        rng = random.Random(seed)
        old_null = np.array([
            truncated_distance(
                baseline.mallows_sample_insertion(d, q, rng),
                baseline.mallows_sample_insertion(d, q, rng), p)
            for _ in range(n_trials)
        ])
        new_null = baseline.compute_rbo_baseline_simplified(
            d=d, r=d, p=p, q=q, n_trials=n_trials, seed=seed, return_std=True)
        self.assertAlmostEqual(new_null["mean"], old_null.mean() - p**d)
        self.assertAlmostEqual(new_null["std"], old_null.std())

        rankings = list(itertools.permutations(range(d)))
        a = np.repeat(values_for_rankings([rankings[0]]), len(rankings), axis=0)
        b = values_for_rankings(rankings)
        old = np.array([truncated_distance(rankings[0], rank, p) for rank in rankings])
        new = plot_violin.compute_rbo_distance(a, b, p=p)
        np.testing.assert_allclose(new, old - p**d, atol=1e-15)
        np.testing.assert_array_equal(new > new_null["mean"], old > old_null.mean())

        old_pairs, new_pairs = np.tile(old, (10, 1)), np.tile(new, (10, 1))
        old_b = pairwise_exceedance(old_pairs, old_null.mean())
        new_b = pairwise_exceedance(new_pairs, new_null["mean"])
        self.assertEqual(old_b["n_exceed"], new_b["n_exceed"])
        self.assertEqual(old_b["n_observations"], new_b["n_observations"])

    def test_plotting_baseline_passes_same_full_depth_and_p(self):
        # Avoid running the 20,000-trial plotting wrapper during a unit test.
        calls = []

        def small_null(**kwargs):
            calls.append(kwargs.copy())
            return baseline_function(**{**kwargs, "n_trials": 8})

        baseline_function = baseline.compute_rbo_baseline_simplified
        with patch.object(baseline, "compute_rbo_baseline_simplified", side_effect=small_null):
            result = plot_violin.compute_rbo_baseline(10, q_values=[.3, .5])
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual((call["d"], call["r"], call["p"]), (10, 10, .9))
        expected = [baseline_function(d=10, r=10, p=.9, q=q, n_trials=8)["mean"]
                    for q in [.3, .5]]
        self.assertEqual(result, {"min": min(expected), "max": max(expected)})


if __name__ == "__main__":
    unittest.main()
