# Binary Search Tree Optimisation

An exploratory numerical project investigating how reward distributions behave across depth-based binary search trees.

The project examines how weighting and rounding strategies can improve the distribution of rewards across nodes, with the aim of reducing or eliminating values that fall below a defined threshold.

## Current Results

- Baseline model: 27 of 100 nodes below threshold
- Weighting optimisation reduced this to 16 of 100 nodes
- Rounding optimisation can move individual cases entirely above threshold
- Derived relationships between tree depth, reward functions and optimal rounding behaviour

## Current Focus

The project is now exploring whether a general optimisation rule can be developed that performs consistently across different seeds and tree configurations. This is currently a work in progress

## Methods

- Python
- Jupyter Notebook
- Numerical optimisation
- Binary search trees
- Weighting strategies
- Rounding analysis
- Parameter exploration
