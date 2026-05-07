---
name: mindquantum
description: "Build, simulate, and analyze quantum circuits with MindQuantum. Provides API patterns, simulator selection, circuit construction, operator algebra, and common pitfalls. Use whenever code imports mindquantum, the user mentions MindQuantum, quantum circuits in Python with MindSpore, or asks about quantum simulation, parameterized quantum circuits, quantum gates, Hamiltonians, or quantum computing in the MindQuantum/MindSpore ecosystem."
---

# MindQuantum — Core API Skill

MindQuantum is a quantum computing framework built on MindSpore. It combines high-performance C++ simulators with MindSpore's automatic differentiation for hybrid quantum-classical computing.

## Environment Setup

Before writing any MindQuantum code, check whether MindQuantum is installed. If the user hits `ModuleNotFoundError: No module named 'mindquantum'` or asks to set up their environment, follow this guide.

### Check Installation

```bash
python -c "import mindquantum; print(mindquantum.__version__)"
```

### Install MindQuantum

```bash
# Recommended: pip install (Python 3.9-3.12 required)
pip install mindquantum
```

MindQuantum's core (circuits, gates, operators, simulators) works **standalone** — no MindSpore required. MindSpore is only needed for the `mindquantum.framework` module (`MQLayer`, hybrid training).

### Optional Dependencies

| Package | When Needed | Install |
|---------|------------|---------|
| MindSpore | `MQLayer` hybrid quantum-classical training | `pip install mindspore` |
| OpenFermion + PySCF | Quantum chemistry (`generate_uccsd`, molecular data) | `pip install openfermion openfermionpyscf` |
| PyTorch + CUDA | QAIA GPU acceleration | `pip install torch` (with CUDA) |

### Verify Installation

```python
# Minimal verification
import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, CNOT
from mindquantum.simulator import Simulator

circ = Circuit().h(0).x(1, 0)
sim = Simulator("mqvector", 2)
sim.apply_circuit(circ)
print(sim.get_qs(ket=True))
# Expected: √2/2¦00⟩ + √2/2¦11⟩
```

### Platform Notes

- **Linux x86_64**: Full support including GPU backends (`mqvector_gpu`, `mqvector_cq` with CUDA 11+)
- **Linux aarch64**: CPU backends only
- **macOS (x86_64 / Apple Silicon)**: CPU backends only
- **Windows x86_64**: CPU backends only
- **Source build**: For unsupported platforms — `git clone https://atomgit.com/mindspore/mindquantum.git && cd mindquantum && python setup.py install`

## Quick Start Pattern

```python
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RX, RY, CNOT, Measure
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator

# Build circuit
circ = Circuit()
circ += H.on(0)
circ += CNOT.on(1, 0)  # target=1, control=0
circ += RX("theta").on(0)  # parameterized gate

# Simulate
sim = Simulator("mqvector", 2)
sim.apply_circuit(circ, pr={"theta": 0.5})
print(sim.get_qs(ket=True))

# Sample
circ += Measure().on(0)
circ += Measure().on(1)
sim.reset()
result = sim.sampling(circ, pr={"theta": 0.5}, shots=1000)
```

## Module Map

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `core.circuit` | Circuit construction | `Circuit`, `UN`, `apply`, `dagger`, `qft` |
| `core.gates` | 50+ quantum gates | `H`, `X`, `RX`, `RY`, `RZ`, `CNOT`, `SWAP`, `U3`, `FSim`, `Measure` |
| `core.operators` | Operator algebra | `QubitOperator`, `FermionOperator`, `Hamiltonian`, `TimeEvolution` |
| `core.parameterresolver` | Symbolic parameters | `ParameterResolver` |
| `simulator` | Simulation backends | `Simulator`, `get_supported_simulator()` |
| `algorithm.nisq` | NISQ ansatz catalog | `HardwareEfficientAnsatz`, `UCCAnsatz`, `QAOAAnsatz`, `StronglyEntangling` |
| `algorithm.compiler` | Circuit decomposition / DAG tools | `DAGCircuit`, `decompose` submodule, compiler rules |
| `algorithm.mapping` | Qubit mapping | `SABRE` |
| `algorithm.qaia` | Quantum-inspired optimization | `SimCIM`, `ASB`, `BSB`, `DSB`, `LQA` |
| `framework` | MindSpore integration | `MQLayer`, `MQAnsatzOnlyLayer`, `MQN2Ops` |
| `io` | Import/export | `OpenQASM`, `HiQASM`, `QCIS` |

## Simulator Selection Guide

Choose the backend based on your task:

| Backend | When to Use | Memory | Notes |
|---------|------------|--------|-------|
| `mqvector` | Default. Pure-state simulation | O(2ⁿ) | All features, gradients, noise (Monte Carlo) |
| `mqvector_gpu` | >8 qubits on NVIDIA GPU | O(2ⁿ) | Requires CUDA 11+ |
| `mqmatrix` | Density matrix / mixed states | O(4ⁿ) | Native noise channels, entropy, purity |
| `mqmps` | Low-entanglement circuits | O(nχ²) | Approximate for high entanglement |
| `stabilizer` | Clifford-only circuits | O(n²) | No parameterized gates |

**Precision rule of thumb:**

- Quantum chemistry → `complex128` (chemical accuracy needs double precision)
- Quantum ML → `complex64` (saves memory, QML is less precision-sensitive)

**Performance tip:** Reuse simulators via `sim.reset()` instead of creating new instances.

## Critical Patterns

### Gate Placement: `.on(target, control)`

```python
X.on(1, 0)  # CNOT: target=1, control=0
X.on(0, [1, 2])  # Toffoli: target=0, controls=[1,2]
H.on(0)  # Single-qubit gate on qubit 0
```

### Parameterized Gates

```python
RX("alpha").on(0)  # named parameter
RX({"alpha": 2}).on(0)  # scaled: rotation = 2*alpha
RX(1.5).on(0)  # fixed value (not trainable)
```

### Circuit Composition

```python
circ1 + circ2  # concatenate
circ * 3  # repeat 3 times
circ.hermitian()  # adjoint / inverse
dagger(circ)  # same as hermitian()
UN(H, range(5))  # apply H to qubits 0-4
```

### Encoder vs Ansatz

This is MindQuantum's key pattern for hybrid quantum-classical models:

```python
encoder = Circuit().rx("x0", 0).ry("x1", 1)
encoder.as_encoder()  # marks params as data-encoding (not trainable)

ansatz = Circuit().ry("w0", 0).x(1, 0).ry("w1", 1)
ansatz.as_ansatz()  # marks params as trainable (default)

full_circuit = encoder + ansatz
```

## When to Use Other Skills

| If your task involves... | Use skill |
|--------------------------|-----------|
| VQE, QAOA, QML, gradient-based training, MindSpore MQLayer | **mq-variational-training** |
| Noise channels, noisy circuits, density matrix, ChannelAdder | **mq-noisy-simulation** |
| QAIA solvers (SimCIM, SB), Ising/Max-Cut optimization | **mq-qaia-solver** |
| Compiling to hardware, qubit mapping, gate decomposition | **mq-circuit-compiler** |
| Molecular Hamiltonians, fermion transforms, UCCSD, mqchem | **mq-quantum-chemistry** |

## Reference Files

Read these on demand when you need deeper API detail:

| File | When to Read |
|------|-------------|
| `reference/circuit-and-gates.md` | Building circuits, gate catalog, controlled gates, circuit operations |
| `reference/parameter-resolver.md` | ParameterResolver algebra, encoder/ansatz marking, parameter manipulation |
| `reference/operators-hamiltonian.md` | QubitOperator, FermionOperator, Hamiltonian, TimeEvolution, commutator |
| `reference/simulator-backends.md` | Simulator API, state operations, sampling, expectation, gradient ops |
| `reference/io-and-visualization.md` | OpenQASM import/export, SVG rendering, circuit printing |

## Common Pitfalls

1. **Endianness**: MindQuantum uses **little-endian** — qubit 0 is the rightmost (least significant) bit in state vectors and measurement results.

2. **`get_expectation_with_grad` requires encoder+ansatz split**: If your circuit has no `as_encoder()` call, all params are treated as ansatz params. Encoder data must be a 2D array `[batch_size, n_encoder_params]`.

3. **MindSpore context**: Always set `ms.set_context(mode=ms.PYNATIVE_MODE)` and `ms.set_device("CPU")` before using `MQLayer`. Graph mode is not supported.

4. **Simulator state persistence**: `apply_circuit` and `apply_gate` **modify** the simulator state. Use `sim.reset()` to return to |0⟩. `sampling` does not change state, but it samples by applying the provided circuit to the simulator's current state.

5. **Noise via Monte Carlo**: When using noise channels with `mqvector`, each call to `sampling` runs Monte Carlo trajectories. Results are statistical — use enough shots. For exact noise simulation, use `mqmatrix` (density matrix) but note O(4ⁿ) memory.

6. **`UN` requires a list**: `UN(H, 5)` is wrong — use `UN(H, range(5))` or `UN(H, [0,1,2,3,4])`.
