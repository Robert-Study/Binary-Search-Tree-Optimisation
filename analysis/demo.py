"""Find the narrowest positive-reward root interval and maximise its mean."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.bst_rewards import reward_matrix, optimise_mixture, minimum_positive_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--reward-floor', type=float, default=1e-6)
    parser.add_argument('--output', type=Path, default=Path('outputs/demo'))
    args = parser.parse_args()
    matrix = reward_matrix(range(1, args.nodes + 1), args.nodes)
    result = minimum_positive_range(matrix, args.reward_floor)
    full = optimise_mixture(matrix)
    first, last = result.first_root, result.last_root
    weights = np.zeros(args.nodes)
    weights[first - 1:last] = result.mixture.weights
    expected = weights @ matrix
    keys = np.arange(1, args.nodes + 1)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output/'root-weights.csv', np.column_stack([keys, weights]),
               delimiter=',', header='root,probability', comments='')
    np.savetxt(args.output/'key-rewards.csv', np.column_stack([keys, expected]),
               delimiter=',', header='key,expected_reward', comments='')
    active = int(np.count_nonzero(weights > 1e-9))
    summary = {
        'nodes': args.nodes, 'root_interval': [first, last],
        'interval_positions': last-first+1, 'roots_with_positive_weight': active,
        'reward_floor': args.reward_floor, 'minimum_expected_reward': float(expected.min()),
        'mean_expected_reward': float(expected.mean()),
        'keys_below_or_equal_to_zero': int(np.count_nonzero(expected <= 0)),
        'weight_sum': float(weights.sum()),
        'feasible_minimum_intervals': result.feasible_intervals,
        'best_minimum_reward_one_position_shorter': result.shorter_range_best_minimum,
        'all_roots_maximin': {'minimum': full.minimum_reward,
                             'mean': float(full.expected_rewards.mean())},
        'scope': 'Smallest contiguous interval, then largest mean subject to the stated positive reward floor; fixed median-split tree family.'
    }
    if args.nodes == 100:
        uniform = matrix[37:62].mean(axis=0)
        summary['uniform_roots_38_62'] = {'minimum': float(uniform.min()),
            'mean': float(uniform.mean()), 'keys_below_or_equal_to_zero': int(np.count_nonzero(uniform <= 0))}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
    if args.nodes == 100:
        axes[0].plot(keys, uniform, color='#b17866', linewidth=1, alpha=.65,
                     label='Uniform roots 38–62')
    axes[0].plot(keys, expected, color='#176b61', linewidth=1.5,
                 label=f'Mean-optimal mixture, roots {first}–{last}')
    axes[0].axhline(0, color='#555555', linewidth=.8)
    axes[0].set(xlabel='Key', ylabel='Expected reward', xlim=(.5,args.nodes+.5),
                title=f'{args.nodes} keys with positive expected reward | mean {expected.mean():.6f}')
    axes[0].legend(frameon=False, loc='upper right')
    axes[1].axvspan(first-.5, last+.5, color='#176b61', alpha=.06)
    axes[1].bar(keys, weights, color='#176b61', width=.85)
    axes[1].set(xlabel='Root key', ylabel='Selection probability', xlim=(.5,args.nodes+.5),
                title=f'Smallest contiguous range: {first}–{last} | {last-first+1} positions, {active} nonzero weights')
    axes[1].text(.01,.96,f'Minimum expected reward: {expected.min():.2e}',
                 transform=axes[1].transAxes,va='top',fontsize=10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    fig.savefig(args.output/'reward-comparison.png',dpi=160)
    plt.close(fig)
    (args.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
