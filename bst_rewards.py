"""Deterministic BST construction and maximin mixtures of root choices.

    depth 1 -> reward 5; in general reward = 6 - depth.

The word 'seed' in the original notebook meant the root key, not an RNG seed.
This model optimises reward allocation, not lookup time or tree balancing.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


def tree_depths(root, n=100):
    """Construct the original median-split tree over keys 1 through n.

    After the root, choose the middle key furthest from its parent; ties choose
    the upper middle key. Return one-based depths in key order.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer.')
    if not isinstance(root, (int, np.integer)) or not 1 <= root <= n:
        raise ValueError('root must be an integer in 1..n.')
    depths = np.zeros(n, dtype=int)
    pending = [(1, n, None, 1)]
    while pending:
        low, high, parent, depth = pending.pop()
        if low > high:
            continue
        if parent is None:
            node = root
        else:
            lower = (low + high) // 2
            upper = (low + high + 1) // 2
            node = lower if abs(lower-parent) > abs(upper-parent) else upper
        depths[node-1] = depth
        pending.extend([(low, node-1, node, depth+1), (node+1, high, node, depth+1)])
    return depths


def reward_matrix(roots, n=100):
    roots = list(roots)
    if not roots:
        raise ValueError('Provide at least one candidate root.')
    return np.array([6 - tree_depths(root, n) for root in roots], dtype=float)


@dataclass
class MixtureResult:
    weights: np.ndarray
    expected_rewards: np.ndarray
    minimum_reward: float
    success: bool


def optimise_mixture(rewards):
    """Maximise the minimum expected key reward over the supplied fixed trees.

    The LP solves max t subject to R.T @ w >= t, sum(w)=1 and w>=0.
    Its optimum applies only to these candidate trees and the stated reward rule.
    """
    matrix = np.asarray(rewards, float)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.all(np.isfinite(matrix)):
        raise ValueError('rewards must be a nonempty finite matrix: trees by keys.')
    trees, keys = matrix.shape
    result = linprog(
        np.r_[np.zeros(trees), -1.0],
        A_ub=np.column_stack([-matrix.T, np.ones(keys)]), b_ub=np.zeros(keys),
        A_eq=np.array([np.r_[np.ones(trees), 0.0]]), b_eq=[1.0],
        bounds=[(0.0, 1.0)]*trees + [(None, None)], method='highs',
    )
    if not result.success:
        raise RuntimeError('Mixture optimisation failed: ' + result.message)
    weights = result.x[:-1]
    expected = weights @ matrix
    return MixtureResult(weights, expected, float(expected.min()), True)


@dataclass
class RangeResult:
    first_root: int
    last_root: int
    mixture: MixtureResult
    reward_floor: float
    shorter_range_best_minimum: float | None
    feasible_intervals: list[tuple[int, int]]


def maximise_mean(rewards, reward_floor=1e-6):
    """Maximise mean reward while every key receives at least reward_floor.

    A specified positive floor makes the strict-positivity problem closed.
    Without it, the best mean may only be a supremum approached at zero reward.
    """
    matrix = np.asarray(rewards, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.all(np.isfinite(matrix)):
        raise ValueError('rewards must be a nonempty finite matrix: trees by keys.')
    if not np.isfinite(reward_floor) or reward_floor <= 0:
        raise ValueError('reward_floor must be finite and strictly positive.')
    result = linprog(
        -matrix.mean(axis=1), A_ub=-matrix.T,
        b_ub=np.full(matrix.shape[1], -reward_floor),
        A_eq=np.ones((1, len(matrix))), b_eq=[1.0], bounds=(0.0, None),
        method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                 'dual_feasibility_tolerance': 1e-9},
    )
    if not result.success:
        raise ValueError('No feasible mean-reward mixture: ' + result.message)
    expected = result.x @ matrix
    if expected.min() < reward_floor - 1e-8 or np.any(expected <= 0):
        raise RuntimeError('The returned mixture fails the positive-reward constraint.')
    return MixtureResult(result.x, expected, float(expected.min()), True)


def minimum_positive_range(rewards, reward_floor=1e-6):
    """Minimise contiguous candidate range, then maximise mean reward.

    Rows must be ordered by consecutive root keys starting at one. Feasibility
    is monotone in interval width, since a wider interval can use zero weights
    for extra roots. Binary search therefore checks widths without skipping a
    narrower feasible interval. Every interval at a tested width is examined.
    """
    matrix = np.asarray(rewards, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.all(np.isfinite(matrix)):
        raise ValueError('rewards must be a nonempty finite matrix: roots by keys.')
    if not np.isfinite(reward_floor) or reward_floor <= 0:
        raise ValueError('reward_floor must be finite and strictly positive.')
    if optimise_mixture(matrix).minimum_reward < reward_floor:
        raise ValueError('The full candidate family cannot meet the reward floor.')
    cache = {}

    def assess(width):
        if width not in cache:
            cache[width] = [
                (start, optimise_mixture(matrix[start:start + width]).minimum_reward)
                for start in range(len(matrix) - width + 1)
            ]
        return cache[width]

    low, high = 1, len(matrix)
    while low < high:
        width = (low + high) // 2
        if any(minimum >= reward_floor for _, minimum in assess(width)):
            high = width
        else:
            low = width + 1
    width = low
    feasible = [(start + 1, start + width) for start, minimum in assess(width)
                if minimum >= reward_floor]
    best = None
    for first, last in feasible:
        mixture = maximise_mean(matrix[first - 1:last], reward_floor)
        # Keep the lower interval when mirrored solutions tie numerically.
        if best is None or mixture.expected_rewards.mean() > best[2].expected_rewards.mean() + 1e-10:
            best = (first, last, mixture)
    shorter_best = max(value for _, value in assess(width - 1)) if width > 1 else None
    return RangeResult(*best, reward_floor, shorter_best, feasible)
