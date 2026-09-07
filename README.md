# Binary Search Tree Optimisation

A study of how the choice of root changes the reward received by each key in a binary search tree. The current solution finds the **smallest contiguous root range that gives every key positive expected reward**, then maximises the mean reward within that range.

## Current solution: roots 4–82

For 100 keys, the narrowest feasible range contains **79 root positions**. Roots **4–82** and the mirrored range **19–97** both work. The mean-optimal solution shown here assigns nonzero probability to **40 roots**.

| Root-selection strategy | Root range | Mean expected reward | Minimum expected reward | Keys at or below zero |
| --- | --- | ---: | ---: | ---: |
| Uniform mixture | 38–62 | 0.200000 | −0.560000 | 19 |
| Maximum minimum reward | 1–100 | 0.105679 | +0.028524 | 0 |
| **Smallest range, then maximum mean** | **4–82** | **0.131744** | **+0.000001** | **0** |

![Expected reward for each key and root-selection probabilities for the minimum-range solution](assets/reward-comparison.png)

The new solution improves the mean by approximately **24.7%** over the all-root maximin mixture while keeping every key's expected reward positive. It trades some protection for the worst-served key for a higher average reward.

[Root probabilities](assets/root-weights.csv) · [Expected rewards by key](assets/key-rewards.csv) · [Search results](assets/results.json)

## How the strategy is found

The keys are 1–100. Once a root is selected, each subtree is split at its middle key; when there are two middle keys, the one further from the parent is chosen. Ties go to the upper middle key. A key at depth d receives reward **6 − d**, with the root at depth one.

A root mixture is a probability distribution over those trees. The search has two ordered objectives:

1. Find the narrowest contiguous candidate-root interval that can keep every expected reward positive.
2. Among intervals of that width, maximise the average expected reward.

For each interval, a linear program finds the largest possible minimum reward. Feasibility is monotone in interval width, so the search can narrow the width efficiently while checking every interval at each tested width.

The best minimum reward across all **78-position intervals is −0.001649**. Since every shorter interval is contained within a 78-position interval, no shorter range can make all keys positive under this tree-construction rule. The 79-position solutions are therefore the numerical minimum for this family.

To optimise the mean with a strict positivity requirement, the final linear program sets a small positive floor:

```text
maximise mean(R.T @ w)
subject to R.T @ w >= 0.000001
           sum(w) = 1
           w >= 0
```

The floor is explicit because a strict inequality by itself can yield a limiting best mean that is approached only as some rewards tend to zero. The reported mean is optimal for the stated floor and minimum-width intervals. “79 positions” describes the span of candidate roots; it does not mean all 79 have nonzero weight.

This is an expected-reward result for the defined tree family. It does not mean that every individual tree rewards every key positively, or that BST lookup complexity has changed.

## Core tests and calculation

```bash
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

The calculation writes the figure, root weights, per-key expectations and search results to `outputs/demo/`. The walkthrough is in [Binary Search Trees.ipynb](Binary%20Search%20Trees.ipynb).

Tests check the tree construction, probability constraints, small problems with known solutions, the minimum feasible range and strict positivity of every expected reward.

[Core tests](tests/test_rewards.py) · [GitHub Actions](https://github.com/Robert-Study/Binary-Search-Tree-Optimisation/actions)

## Further investigation

This solution is implemented. I intend to look for a better strategy by exploring alternative subtree rounding rules and tree constructions, aiming to retain positive rewards with a narrower root range or a higher mean.
