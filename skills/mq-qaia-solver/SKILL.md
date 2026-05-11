---
name: mq-qaia-solver
description: "Solve Ising/QUBO-style combinatorial optimization problems using MindQuantum's Quantum Annealing-Inspired Algorithms (QAIA). Covers SimCIM, ASB/BSB/DSB, TSB/USB/LSB, LQA, CFC, CAC, SFC, and NMFA, with solver-specific CPU/GPU/NPU backend support. This is not circuit-based QAOA: QAIA solvers operate directly on an Ising coupling matrix J and optional field h. Use when the user mentions QAIA, SimCIM, simulated bifurcation, Ising solver, Max-Cut solver, QUBO, or a combinatorial problem already encoded as Ising/QUBO."
---

# QAIA: Quantum Annealing-Inspired Algorithms

MindQuantum's QAIA module provides quantum annealing-inspired solvers for combinatorial optimization problems encoded as Ising models. These solvers do not build quantum circuits; they evolve solver-specific state variables from a coupling matrix.

**Key distinction:** QAIA solvers are NOT circuit-based QAOA. They solve Ising/QUBO problems directly using physics-inspired dynamics. For circuit-based QAOA, use the `mq-variational-training` skill.

## The Ising Model

QAIA's `calc_energy()` uses this Ising energy convention for a symmetric coupling matrix:

$$H(\mathbf{s}) = -\frac{1}{2}\sum_{i,j} J_{ij} s_i s_j - \sum_i h_i s_i$$

where $s_i \in \{-1, +1\}$ are spin variables, $J$ is the coupling matrix, and $h$ is the external field.

## Quick Start

```python
import numpy as np
from scipy.sparse import coo_matrix
from mindquantum.algorithm.qaia import BSB

# Define coupling matrix J (symmetric, from graph edges)
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
n_nodes = 4
row = [e[0] for e in edges] + [e[1] for e in edges]
col = [e[1] for e in edges] + [e[0] for e in edges]
data = [-1] * len(row)
J = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

# Solve
solver = BSB(J, batch_size=100, n_iter=500)
solver.update()

# Results
cuts = solver.calc_cut()
print(f"Best cut value: {max(cuts)}")
spins = np.sign(solver.x)  # Final spin configuration
```

## Available Solvers

| Solver | Import | Name in source docstring | Documented / implemented backends |
|--------|--------|--------------------------|----------------------------------|
| `SimCIM` | `from mindquantum.algorithm.qaia import SimCIM` | Simulated Coherent Ising Machine | `cpu-float32`, `gpu-float32`, `npu-float32` |
| `ASB` | `from mindquantum.algorithm.qaia import ASB` | Adiabatic SB algorithm | `cpu-float32`, `gpu-float32`, `npu-float32` |
| `BSB` | `from mindquantum.algorithm.qaia import BSB` | Ballistic SB algorithm | `cpu-float32`, `gpu-float32`, `gpu-float16`, `gpu-int8`, `npu-float32` |
| `DSB` | `from mindquantum.algorithm.qaia import DSB` | Discrete SB algorithm | `cpu-float32`, `gpu-float32`, `gpu-float16`, `gpu-int8`, `npu-float32` |
| `LQA` | `from mindquantum.algorithm.qaia import LQA` | Local quantum annealing algorithm | `cpu-float32`, `gpu-float32`, `npu-float32` |
| `CAC` | `from mindquantum.algorithm.qaia import CAC` | Coherent Ising Machine with chaotic amplitude control | `cpu-float32`, `gpu-float32`, `npu-float32` |
| `CFC` | `from mindquantum.algorithm.qaia import CFC` | Coherent Ising Machine with chaotic feedback control | `cpu-float32`, `gpu-float32`, `npu-float32` |
| `SFC` | `from mindquantum.algorithm.qaia import SFC` | Coherent Ising Machine with separated feedback control | `cpu-float32`, `gpu-float32`, `npu-float32` |
| `NMFA` | `from mindquantum.algorithm.qaia import NMFA` | Noisy Mean-field Annealing algorithm | `cpu-float32`, `gpu-float32`, `npu-float32` |
| `TSB` | `from mindquantum.algorithm.qaia import TSB` | Ternary Simulated Bifurcation algorithm | `cpu-float32`, `gpu-float32` |
| `USB` | `from mindquantum.algorithm.qaia import USB` | Uniformly Quantized Simulated Bifurcation | `cpu-float32`, `gpu-float32` |
| `LSB` | `from mindquantum.algorithm.qaia import LSB` | Logarithmic Quantized Simulated Bifurcation | `cpu-float32`, `gpu-float32` |

## Core API

QAIA solvers share this core constructor and method pattern; individual classes may expose additional parameters such as `dt`, `xi`, `threshold`, or `strategy`:

```python
solver = SolverClass(
    J,  # Coupling matrix: numpy.array or scipy.sparse
    h=None,  # External field: numpy.array [N, 1] (optional)
    x=None,  # Initial spins: numpy.array [N, batch_size] (optional)
    n_iter=1000,  # Number of iterations
    batch_size=1,  # Parallel samples
    backend="cpu-float32",  # Compute backend; supported values are solver-specific
)

solver.update()  # Run the optimization dynamics
solver.calc_cut()  # Max-Cut value for each batch (returns array)
solver.calc_energy()  # Ising energy for each batch
spins = np.sign(solver.x)  # Extract discrete spin assignments
```

## Backend Notes

Backends are not uniform across all QAIA classes.

- `gpu-float32` uses PyTorch CUDA in the Python implementations.
- `npu-float32` uses `torch_npu` and is only implemented by the non-quantized solvers listed above.
- `gpu-float16` and `gpu-int8` are only implemented in `BSB` and `DSB`; their source note warns that `gpu-int8` may not perform well on dense graphs or graphs with continuous coefficients.
- `TSB`, `USB`, and `LSB` explicitly validate only `cpu-float32` and `gpu-float32`.

```python
solver = BSB(J, batch_size=1000, n_iter=2000, backend="gpu-float32")
solver.update()
```

## Max-Cut Workflow (Complete)

```python
import numpy as np
from scipy.sparse import coo_matrix
from mindquantum.algorithm.qaia import BSB, DSB, SimCIM


# 1. Load graph (e.g., GSet format)
def load_gset(filepath):
    """Load GSet benchmark graph as sparse coupling matrix."""
    import pandas as pd

    data = pd.read_csv(filepath, sep=" ", header=0)
    n = int(data.columns[0])
    rows = np.concatenate([data.iloc[:, 0] - 1, data.iloc[:, 1] - 1])
    cols = np.concatenate([data.iloc[:, 1] - 1, data.iloc[:, 0] - 1])
    vals = np.concatenate([data.iloc[:, 2], data.iloc[:, 2]])
    return coo_matrix((-vals, (rows, cols)), shape=(n, n))


# 2. Try multiple solvers
J = load_gset("G22.txt")  # 2000-node graph

results = {}
for name, Solver in [("BSB", BSB), ("DSB", DSB), ("SimCIM", SimCIM)]:
    solver = Solver(J, batch_size=100, n_iter=1000)
    solver.update()
    best_cut = max(solver.calc_cut())
    results[name] = best_cut
    print(f"{name}: MaxCut = {best_cut}")

# 3. Top result in this run
top = max(results, key=results.get)
print(f"Top solver in this run: {top} with cut = {results[top]}")
```

## Problem Formulation Guide

### Max-Cut → Ising

For Max-Cut, negate the adjacency matrix:

```python
# J[i,j] = -weight(i,j) for edges, 0 otherwise
# h = None (no external field)
```

### QUBO → Ising

Convert Quadratic Unconstrained Binary Optimization. This helper assumes the common upper-triangular convention
`E(x) = sum_i Q[i,i] x_i + sum_{i<j} Q[i,j] x_i x_j` and returns `(J, h, constant)` for the QAIA energy
`H(s) = -sum_{i<j} J[i,j] s_i s_j - sum_i h[i] s_i + constant`.

```python
def qubo_to_ising(Q):
    """Convert upper-triangular QUBO matrix Q to QAIA Ising (J, h, constant)."""
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    constant = 0.0
    for i in range(n):
        h[i] -= Q[i, i] / 2
        constant += Q[i, i] / 2
        for j in range(i + 1, n):
            qij = Q[i, j]
            J[i, j] = -qij / 4
            J[j, i] = -qij / 4
            h[i] -= qij / 4
            h[j] -= qij / 4
            constant += qij / 4
    return J, h.reshape(-1, 1), constant
```

### Graph Coloring, SAT, TSP

Encode constraints as penalty terms in the Ising Hamiltonian. The coupling matrix $J$ and field $h$ encode both the objective and constraints.

## Parameter Notes

| Parameter | Effect | Guidance |
|-----------|--------|----------|
| `batch_size` | Number of parallel samples | Increases the number of independent candidates returned by one solver run. |
| `n_iter` | Evolution steps | Controls how many update steps `update()` performs. |
| `backend` | Compute device / precision | Must be chosen from the backends supported by that solver class. |

### Comparing Solvers

The source tree documents multiple algorithms but does not provide a universal ranking by graph size, density, or hardness. When solution quality matters, run the candidate solvers with the same `J`, `h`, `batch_size`, `n_iter`, and random seed policy, then compare `calc_energy()` or the problem-specific objective.

## Important Notes

1. **In-place modification:** `solver.x` is modified during `update()`. Pass `x.copy()` if you need the original.
2. **Sparse matrices:** Use `scipy.sparse` for large graphs — solvers accept both dense and sparse formats.
3. **Symmetry:** $J$ must be symmetric. If your adjacency matrix is asymmetric, symmetrize: `J = (J + J.T) / 2`.
4. **Sign convention:** QAIA energy uses $H = -\frac{1}{2}\sum_{i,j} J_{ij}s_is_j - \sum_i h_i s_i$. For Max-Cut examples in this repository, edge weights are negated before constructing `J`.
