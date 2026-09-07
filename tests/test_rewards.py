import unittest
import numpy as np

from bst_rewards import tree_depths, reward_matrix, optimise_mixture, maximise_mean, minimum_positive_range


class RewardTests(unittest.TestCase):
    def test_small_balanced_tree_matches_manual_depths(self):
        np.testing.assert_array_equal(tree_depths(4, 7), [3, 2, 3, 1, 3, 2, 3])

    def test_every_key_appears_once_at_valid_depth(self):
        for n in [1, 2, 7, 100]:
            for root in range(1, n+1):
                depths = tree_depths(root, n)
                self.assertEqual(np.count_nonzero(depths == 1), 1)
                self.assertEqual(depths[root-1], 1)
                self.assertTrue(np.all((depths >= 1) & (depths <= n)))

    def test_symmetric_game_has_known_optimum(self):
        result = optimise_mixture([[1, -1], [-1, 1]])
        np.testing.assert_allclose(result.weights, [.5, .5], atol=1e-8)
        self.assertAlmostEqual(result.minimum_reward, 0)

    def test_feasible_mixture_cannot_be_worse_than_uniform(self):
        matrix = reward_matrix(range(38, 63))
        result = optimise_mixture(matrix)
        self.assertAlmostEqual(result.weights.sum(), 1)
        self.assertTrue(np.all(result.weights >= -1e-9))
        self.assertGreaterEqual(result.minimum_reward+1e-8, matrix.mean(axis=0).min())

    def test_invalid_root_is_rejected(self):
        for root in [0, 101, 1.5]:
            with self.assertRaises(ValueError):
                tree_depths(root)

    def test_full_candidate_family_has_a_positive_reward_witness(self):
        matrix = reward_matrix(range(1, 101))
        result = optimise_mixture(matrix)
        self.assertTrue(np.all(result.weights @ matrix > .028))
        self.assertAlmostEqual(result.weights.sum(), 1)


    def test_mean_objective_has_known_solution(self):
        result = maximise_mean([[3, -1], [-1, 2]], reward_floor=.1)
        self.assertAlmostEqual(result.weights[0], 1.9 / 3)
        self.assertAlmostEqual(result.minimum_reward, .1)

    def test_interval_search_on_small_known_problem(self):
        result = minimum_positive_range([[2, -1], [-1, 2], [-4, -4]], .1)
        self.assertEqual((result.first_root, result.last_root), (1, 2))
        self.assertLess(result.shorter_range_best_minimum, 0)

    def test_narrowest_100_key_range_and_strict_positivity(self):
        matrix = reward_matrix(range(1, 101))
        result = minimum_positive_range(matrix)
        self.assertEqual((result.first_root, result.last_root), (4, 82))
        self.assertEqual(result.feasible_intervals, [(4, 82), (19, 97)])
        self.assertLess(result.shorter_range_best_minimum, 0)
        self.assertTrue(np.all(result.mixture.expected_rewards > 0))
        self.assertAlmostEqual(result.mixture.weights.sum(), 1)
        self.assertGreater(result.mixture.expected_rewards.mean(), .1317)

    def test_impossible_positive_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            minimum_positive_range([[1, -1], [-1, 1]])

if __name__ == '__main__':
    unittest.main()
