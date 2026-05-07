---
name: mq-qaia-solver
description: "Solve combinatorial optimization problems using MindQuantum's Quantum Annealing-Inspired Algorithms (QAIA). Supports SimCIM, Simulated Bifurcation (ASB/BSB/DSB), LQA, CFC, CAC, SFC, NMFA with CPU/GPU/NPU backends. Use for Ising model optimization, Max-Cut, SAT, QUBO problems, and large-scale combinatorial optimization. This is NOT circuit-based QAOA — these are classical algorithms inspired by quantum annealing that run on CPU/GPU without a quantum circuit. Use when the user mentions QAIA, SimCIM, simulated bifurcation, Ising solver, combinatorial optimization without circuits, Max-Cut solver, or wants to solve optimization on GPU/NPU."
---

# QAIA: Quantum Annealing-Inspired Algorithms

MindQuantum's QAIA module provides classical algorithms that simulate quantum mechanical principles to solve combinatorial optimization problems. These run entirely on CPU/GPU/NPU — no quantum circuit required.

**Key distinction:** QAIA solvers are NOT circuit-based QAOA. They solve Ising/QUBO problems directly using physics-inspired dynamics. For circuit-based QAOA, use the `mq-variational-training` skill.

## The Ising Model

All QAIA solvers minimize the Ising Hamiltonian:

$$H(\mathbf{s}) = -\sum_{i,j} J_{ij} s_i s_j - \sum_i h_i s_i$$

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

| Solver | Import | Physical Inspiration | Strengths |
|--------|--------|---------------------|-----------|
| `SimCIM` | `from mindquantum.algorithm.qaia import SimCIM` | Coherent Ising Machine | Good for dense graphs |
| `ASB` | `from mindquantum.algorithm.qaia import ASB` | Adiabatic Simulated Bifurcation | Smooth convergence |
| `BSB` | `from mindquantum.algorithm.qaia import BSB` | Ballistic Simulated Bifurcation | Fast, good default |
| `DSB` | `from mindquantum.algorithm.qaia import DSB` | Discrete Simulated Bifurcation | Fast, discrete spins |
| `LQA` | `from mindquantum.algorithm.qaia import LQA` | Local Quantum Annealing | Exploits quantum tunneling |
| `CAC` | `from mindquantum.algorithm.qaia import CAC` | Chaotic Amplitude Control | Escapes local minima |
| `CFC` | `from mindquantum.algorithm.qaia import CFC` | Chaotic Amplitude Feedback | Enhanced exploration |
| `SFC` | `from mindquantum.algorithm.qaia import SFC` | Discrete Amplitude Feedback | Good for structured problems |
| `NMFA` | `from mindquantum.algorithm.qaia import NMFA` | Noisy Mean-Field Annealing | Simple, robust |
| `TSB` | `from mindquantum.algorithm.qaia import TSB` | Tensor-network SB | Large-scale |
| `USB` | `from mindquantum.algorithm.qaia import USB` | Unified SB | Balanced approach |
| `LSB` | `from mindquantum.algorithm.qaia import LSB` | Layered SB | Structured problems |

## Uniform API

All solvers share the same interface:

```python
solver = SolverClass(
    J,  # Coupling matrix: numpy.array or scipy.sparse
    h=None,  # External field: numpy.array [N, 1] (optional)
    x=None,  # Initial spins: numpy.array [N, batch_size] (optional)
    n_iter=1000,  # Number of iterations
    batch_size=1,  # Parallel samples
    backend="cpu-float32",  # Compute backend
)

solver.update()  # Run the optimization dynamics
solver.calc_cut()  # Max-Cut value for each batch (returns array)
solver.calc_energy()  # Ising energy for each batch
spins = np.sign(solver.x)  # Extract discrete spin assignments
```

## GPU/NPU Acceleration

```python
# Available backends
backends = {
    "cpu-float32",  # CPU, 32-bit (default)
    "gpu-float32",  # NVIDIA GPU, 32-bit
    "gpu-float16",  # NVIDIA GPU, 16-bit (fastest)
    "gpu-int8",  # NVIDIA GPU, 8-bit quantized
    "npu-float32",  # Ascend NPU, 32-bit
}

# GPU requires PyTorch with CUDA
solver = BSB(J, batch_size=1000, n_iter=2000, backend="gpu-float32")
solver.update()
```

### Performance Comparison

For a 2000-node graph (GSet G22):

| Backend | Typical Runtime | Notes |
|---------|----------------|-------|
| `cpu-float32` | ~30s | Baseline |
| `gpu-float32` | ~3s | 10x speedup |
| `gpu-float16` | ~1.5s | 20x speedup, slight precision loss |
| `gpu-int8` | ~1s | Fastest, may lose solution quality |

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

# 3. Best result
best = max(results, key=results.get)
print(f"Best solver: {best} with cut = {results[best]}")
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

## Parameter Tuning Guide

| Parameter | Effect | Guidance |
|-----------|--------|----------|
| `batch_size` | Number of parallel random starts | Higher = better chance of global optimum. Start with 100. |
| `n_iter` | Evolution steps | More = better convergence. Start with 1000, increase to 5000 for hard problems. |
| `backend` | Compute device | Use `gpu-float32` for >500-node problems. |

### Solver Selection

| Problem Size | Graph Density | Recommended |
|-------------|---------------|-------------|
| Small (<100 nodes) | Any | `BSB` or `DSB` |
| Medium (100-2000) | Sparse | `DSB` (fast) or `BSB` (quality) |
| Medium (100-2000) | Dense | `SimCIM` or `BSB` |
| Large (>2000) | Sparse | `DSB` on GPU |
| Hard instances | Any | Try `LQA`, `CAC`, or `CFC` for escaping local minima |

## Important Notes

1. **In-place modification:** `solver.x` is modified during `update()`. Pass `x.copy()` if you need the original.
2. **Sparse matrices:** Use `scipy.sparse` for large graphs — solvers accept both dense and sparse formats.
3. **Symmetry:** $J$ must be symmetric. If your adjacency matrix is asymmetric, symmetrize: `J = (J + J.T) / 2`.
4. **Sign convention:** QAIA minimizes $H = -\sum J_{ij} s_i s_j$. For Max-Cut, negate the adjacency weights so minimizing $H$ maximizes the cut.
