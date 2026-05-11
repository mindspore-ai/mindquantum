# MindQuantum Skills

Agent skills for [MindQuantum](https://atomgit.com/mindspore/mindquantum), the quantum computing framework. Install once, then build quantum circuits, train variational algorithms, simulate noisy systems, and solve combinatorial optimization problems.

## Installation

```bash
npx skills add https://atomgit.com/mindspore/mindquantum.git
```

Or install a specific skill:

```bash
npx skills add https://atomgit.com/mindspore/mindquantum.git --skill mindquantum
```

## Skills

| Skill | Description |
|-------|-------------|
| **[mindquantum](mindquantum/)** | Core API: circuit construction, gates, operators, simulator selection, common patterns |
| **[mq-variational-training](mq-variational-training/)** | Train VQE, QAOA, and QML models with gradient-based optimization and MindSpore MQLayer |
| **[mq-noisy-simulation](mq-noisy-simulation/)** | Simulate noisy quantum circuits: noise channels, ChannelAdder, density matrix backends |
| **[mq-qaia-solver](mq-qaia-solver/)** | Solve Ising/QUBO-style problems with QAIA solvers, using solver-specific CPU/GPU/NPU backends |
| **[mq-circuit-compiler](mq-circuit-compiler/)** | Compile circuits for hardware: gate decomposition, SABRE qubit mapping, topology management |
| **[mq-quantum-chemistry](mq-quantum-chemistry/)** | Quantum chemistry VQE: molecular Hamiltonians, fermion transforms, UCCSD ansatz, mqchem |

## Which Skill Do I Need

| I want to... | Skill |
|--------------|-------|
| Build and simulate a quantum circuit | `mindquantum` |
| Train a VQE or QAOA algorithm | `mq-variational-training` |
| Build a quantum classifier (QML) | `mq-variational-training` |
| Simulate noise effects on my circuit | `mq-noisy-simulation` |
| Solve Ising, QUBO, or Max-Cut formulations with QAIA | `mq-qaia-solver` |
| Map or decompose a circuit for a target topology | `mq-circuit-compiler` |
| Compute molecular ground state energies | `mq-quantum-chemistry` |

## Quick Example

After installing, just describe what you want:

> "Build a 4-qubit QAOA circuit for Max-Cut on a triangle graph, optimize with BFGS, and show the solution."

The agent will use the appropriate skills to generate correct, idiomatic MindQuantum code.

## Requirements

- [MindQuantum](https://atomgit.com/mindspore/mindquantum) (`pip install mindquantum`)
- [MindSpore](https://www.mindspore.cn/install) (optional, for `MQLayer` hybrid training)
- [OpenFermion](https://github.com/quantumlib/OpenFermion) + [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF) (optional, for quantum chemistry)
