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
