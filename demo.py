"""Compare uniform and maximin root mixtures for a deterministic BST family."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bst_rewards import reward_matrix, optimise_mixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--root-min', type=int, default=38)
    parser.add_argument('--root-max', type=int, default=62)
    parser.add_argument('--output', type=Path, default=Path('outputs/demo'))
    args = parser.parse_args()
    if not 1 <= args.root_min <= args.root_max <= args.nodes:
        parser.error('Require 1 <= root-min <= root-max <= nodes')
    roots = np.arange(args.root_min, args.root_max+1)
    matrix = reward_matrix(roots, args.nodes)
    uniform = matrix.mean(axis=0)
    result = optimise_mixture(matrix)
    expanded = optimise_mixture(reward_matrix(range(1, args.nodes+1), args.nodes))
    args.output.mkdir(parents=True, exist_ok=True)
    keys = np.arange(1, args.nodes+1)
    np.savetxt(args.output / 'root-weights.csv', np.column_stack([roots, result.weights]),
               delimiter=',', header='root,weight', comments='')
    np.savetxt(args.output / 'key-rewards.csv', np.column_stack([keys, uniform, result.expected_rewards]),
               delimiter=',', header='key,uniform_reward,maximin_reward', comments='')
    np.savetxt(args.output / 'all-root-weights.csv', np.column_stack([keys, expanded.weights]),
               delimiter=',', header='root,weight', comments='')
    np.savetxt(args.output / 'expanded-key-rewards.csv', np.column_stack([keys, expanded.expected_rewards]),
               delimiter=',', header='key,expected_reward', comments='')
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ax.plot(keys, uniform, '.-', color='#557d8a', linewidth=.6, label=f'Uniform, roots {args.root_min}–{args.root_max}')
    ax.plot(keys, result.expected_rewards, '.-', color='#b25a30', linewidth=.6, label=f'Maximin, roots {args.root_min}–{args.root_max}')
    ax.plot(keys, expanded.expected_rewards, '-', color='#19664a', linewidth=1.2, label='Maximin over all roots')
    ax.axhline(0, color='#777777', linewidth=.8)
    ax.set(xlabel='Key', ylabel='Expected reward', title=f'BST reward allocation | {args.nodes} keys')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(args.output / 'reward-comparison.png', dpi=150)
    plt.close(fig)
    summary = {'nodes': args.nodes, 'roots': [args.root_min, args.root_max], 'threshold': 0,
               'negative_tolerance': 1e-9,
               'uniform': {'minimum_reward': float(uniform.min()), 'mean_reward': float(uniform.mean()),
                           'keys_below_zero': int(np.count_nonzero(uniform < -1e-9))},
               'maximin': {'minimum_reward': result.minimum_reward, 'mean_reward': float(result.expected_rewards.mean()),
                           'keys_below_zero': int(np.count_nonzero(result.expected_rewards < -1e-9))},
               'expanded_maximin': {'roots': [1, args.nodes], 'minimum_reward': expanded.minimum_reward,
                                    'mean_reward': float(expanded.expected_rewards.mean()),
                                    'keys_below_zero': int(np.count_nonzero(expanded.expected_rewards < -1e-9)),
                                    'all_keys_strictly_positive': bool(np.all(expanded.expected_rewards > 1e-9))},
               'scope': 'Numerical optimum over the fixed candidate trees, not all possible BST structures.'}
    (args.output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
