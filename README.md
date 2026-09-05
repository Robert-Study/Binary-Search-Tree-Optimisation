# Binary Search Tree Optimisation

An exploratory numerical project investigating how **depth-based reward distributions** behave across binary search trees and how weighting or rounding strategies can improve outcomes across different tree configurations.

The notebook combines simulation, optimisation and interactive visualisation to examine how node position and tree depth affect rewards across a range of seeds.

> **Status:** exploratory / work in progress

---

## Current Results

- Baseline configuration: **27 of 100 nodes** fall below the target threshold
- Weighting optimisation reduces this to **16 of 100 nodes**
- Targeted rounding rules can move individual configurations entirely above threshold
- Seeds **38–62** are explored as a candidate region for improved average reward behaviour in the default setup
- The analysis examines relationships between tree depth, node position, weighting and rounding behaviour

---

## What the Notebook Does

- Generates reward distributions for binary search trees across multiple seeds
- Builds and visualises trees using **NetworkX**
- Compares rewards for individual seeds and weighted averages
- Produces tabular comparisons across nodes and configurations
- Provides an interactive seed selector using **ipywidgets**
- Tests weighting and rounding adjustments
- Uses numerical optimisation tools from **SciPy** to explore improved configurations

---

## Current Focus

The next objective is to determine whether the improvements can be expressed as a **general optimisation rule** that performs consistently across different seeds and tree structures, rather than relying on configuration-specific adjustments.

---

## Technical Stack

`Python` · `NumPy` · `Pandas` · `SciPy` · `Matplotlib` · `NetworkX` · `Jupyter` · `ipywidgets`

**Methods:** binary search trees · numerical optimisation · parameter exploration · reward weighting · rounding analysis · interactive visualisation

---

## Repository

```text
Binary-Search-Tree-Optimisation/
├── Binary Search Trees.ipynb
└── README.md
```

This repository is deliberately presented as an **exploratory project** rather than a finished optimisation result; the emphasis is on the modelling process, visual analysis and development of a more general solution.
