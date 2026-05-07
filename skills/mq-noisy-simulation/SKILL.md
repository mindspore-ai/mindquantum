---
name: mq-noisy-simulation
description: "Simulate noisy quantum circuits with MindQuantum. Covers noise channels (depolarizing, amplitude damping, phase damping, thermal relaxation, Kraus), the ChannelAdder system for systematic noise insertion, NoiseBackend for automatic noise injection, and density matrix simulation with mqmatrix. Use whenever the user mentions noise, decoherence, error rates, noise models, noisy simulation, density matrix, mixed states, ChannelAdder, quantum error channels, fidelity under noise, or wants to study how noise affects quantum circuits or algorithms."
---

# Noisy Quantum Circuit Simulation

MindQuantum provides two approaches to noise simulation:

1. **Monte Carlo trajectories** — add noise channels to circuits, sample via `mqvector`. Each shot randomly applies or skips the noise gate. Fast, scales to many qubits, but results are statistical.
2. **Density matrix** — use `mqmatrix` backend for exact mixed-state evolution. Deterministic but O(4ⁿ) memory — practical for roughly ≤12 qubits on ordinary machines.

## Approach 1: Manual Noise Channels

Add noise gates directly into your circuit like any other gate:

```python
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import (
    H,
    CNOT,
    RX,
    Measure,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
    BitFlipChannel,
    PauliChannel,
    ThermalRelaxationChannel,
)
from mindquantum.simulator import Simulator

# Build noisy circuit
circ = Circuit()
circ += H.on(0)
circ += DepolarizingChannel(0.01).on(0)  # 1% depolarizing after H
circ += CNOT.on(1, 0)
circ += DepolarizingChannel(0.02).on(0)  # 2% after CNOT (per qubit)
circ += DepolarizingChannel(0.02).on(1)
circ += Measure().on(0)
circ += Measure().on(1)

# Simulate via Monte Carlo sampling
sim = Simulator("mqvector", 2)
result = sim.sampling(circ, shots=10000)
print(result.data)  # {'00': 4980, '11': 4720, '01': 150, '10': 150}
result.svg()  # Visualize histogram
```

### Available Noise Channels

| Channel | Constructor | Physical Model |
|---------|-------------|---------------|
| `BitFlipChannel` | `BitFlipChannel(p)` | X gate with probability p |
| `PhaseFlipChannel` | `PhaseFlipChannel(p)` | Z gate with probability p |
| `BitPhaseFlipChannel` | `BitPhaseFlipChannel(p)` | Y gate with probability p |
| `DepolarizingChannel` | `DepolarizingChannel(p)` | Random X/Y/Z each with p/3 |
| `PauliChannel` | `PauliChannel(px, py, pz)` | Custom Pauli probabilities |
| `AmplitudeDampingChannel` | `AmplitudeDampingChannel(γ)` | Energy decay (T1 process) |
| `PhaseDampingChannel` | `PhaseDampingChannel(γ)` | Dephasing (T2 process) |
| `ThermalRelaxationChannel` | `ThermalRelaxationChannel(T1, T2, gate_time)` | Combined T1/T2 relaxation |
| `KrausChannel` | `KrausChannel('name', [K0, K1, ...])` | Arbitrary Kraus operators |
| `GroupedPauliChannel` | `GroupedPauliChannel(probs).on(qubits)` | Batched per-qubit Pauli channels; `probs` has shape `(n_qubits, 3)` |

### Helper: Add Noise After Every Gate

```python
from mindquantum.core.gates import DepolarizingChannel, Measure, NoiseGate


def add_noise_to_circuit(circuit, p_depol=0.01):
    """Insert depolarizing noise after every non-noise, non-measure gate."""
    noisy = Circuit()
    for gate in circuit:
        noisy += gate
        if not isinstance(gate, (Measure, NoiseGate)):
            for q in gate.obj_qubits:
                noisy += DepolarizingChannel(p_depol).on(q)
    return noisy
```

## Approach 2: ChannelAdder System

For systematic, configurable noise injection without manually editing circuits. Uses rules to decide which gates get noise.

```python
from mindquantum.core.circuit.channel_adder import (
    ChannelAdderBase,
    BitFlipAdder,
    DepolarizingChannelAdder,
    MeasureAccepter,
    NoiseExcluder,
    QubitIDConstrain,
    QubitNumberConstrain,
    GateSelector,
    SequentialAdder,
    MixerAdder,
    ReverseAdder,
)
```

### Built-in Adders

| Adder | Purpose |
|-------|---------|
| `BitFlipAdder(p)` | Add BitFlipChannel after matching gates |
| `DepolarizingChannelAdder(p, n_qubits)` | Add DepolarizingChannel |
| `MeasureAccepter` | Select only measurement gates |
| `NoiseExcluder` | Exclude existing noise gates from re-noising |
| `QubitIDConstrain(qubit_ids)` | Only add noise on specific qubits |
| `QubitNumberConstrain(n)` | Only add noise to n-qubit gates |
| `GateSelector(gate_types)` | Only add noise after specific gate types |
| `SequentialAdder([adder1, adder2])` | Apply multiple adders in sequence |
| `MixerAdder([adder1, adder2])` | Add noise only if ALL sub-adders agree |
| `ReverseAdder(adder)` | Flip accept/reject logic |

### Example: Realistic Noise Model

```python
from mindquantum.core.circuit.channel_adder import (
    DepolarizingChannelAdder,
    QubitNumberConstrain,
    MixerAdder,
    SequentialAdder,
)

# Different noise rates for 1-qubit vs 2-qubit gates
single_qubit_noise = MixerAdder(
    [
        DepolarizingChannelAdder(0.001, 1),
        QubitNumberConstrain(1),
    ]
)
two_qubit_noise = MixerAdder(
    [
        DepolarizingChannelAdder(0.01, 2),
        QubitNumberConstrain(2),
    ]
)
noise_model = SequentialAdder([single_qubit_noise, two_qubit_noise])
```

### Custom ChannelAdder

```python
from mindquantum.core.circuit.channel_adder import ChannelAdderBase
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import DepolarizingChannel, Measure, NoiseGate


class QubitSpecificDepolarizing(ChannelAdderBase):
    """Apply different noise rates per qubit."""

    def __init__(self, qubit_id, p):
        self.qubit_id = qubit_id
        self.p = p
        super().__init__()

    def _accepter(self):
        return [lambda g: self.qubit_id in g.obj_qubits or self.qubit_id in g.ctrl_qubits]

    def _excluder(self):
        return [lambda g: isinstance(g, (Measure, NoiseGate))]

    def _handler(self, gate):
        return Circuit([DepolarizingChannel(self.p).on(self.qubit_id)])
```

## Approach 3: NoiseBackend

Wraps a simulator backend to automatically inject noise via a ChannelAdder:

```python
from mindquantum.simulator import Simulator
from mindquantum.simulator.noise import NoiseBackend

# Create noisy simulator
noise_sim = Simulator(NoiseBackend("mqvector", n_qubits, noise_model))

# Use exactly like a normal simulator
result = noise_sim.sampling(circuit, shots=10000)

# Inspect the transformed circuit (with noise inserted)
noisy_circ = noise_sim.backend.transform_circ(circuit)
noisy_circ.svg()  # See where noise channels were added
```

## Approach 4: Density Matrix (mqmatrix)

For exact noise simulation on small systems:

```python
sim = Simulator("mqmatrix", 4)
sim.apply_circuit(noisy_circuit)

# Density matrix operations
rho = sim.get_qs()  # Full density matrix
entropy = sim.entropy()  # Von Neumann entropy
purity = sim.purity()  # Tr(ρ²)
rho_sub = sim.get_partial_trace([0, 1])  # Trace out qubits 0,1
```

### When to Use mqmatrix vs mqvector

| Factor | `mqvector` + Monte Carlo | `mqmatrix` |
|--------|-------------------------|------------|
| Memory | O(2ⁿ) | O(4ⁿ) |
| Max qubits (16 GB) | ~30 | ~14-15 raw memory, ~12 practical |
| Accuracy | Statistical (more shots = better) | Exact |
| Speed per shot | Fast | N/A (single evolution) |
| Mixed-state queries | No | Yes (entropy, purity, partial trace) |
| Gradient support | Yes | Yes (but no `circ_left` / `simulator_left`) |

**Rule of thumb:** Use `mqmatrix` for ≤12 qubits when you need exact mixed-state properties. Use `mqvector` with Monte Carlo for larger systems.

## Noisy VQE Example

```python
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY, CNOT, DepolarizingChannel
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator
import numpy as np
from scipy.optimize import minimize

# Noisy ansatz
ansatz = Circuit()
ansatz += RY("a0").on(0)
ansatz += DepolarizingChannel(0.005).on(0)
ansatz += RY("a1").on(1)
ansatz += DepolarizingChannel(0.005).on(1)
ansatz += CNOT.on(1, 0)
ansatz += DepolarizingChannel(0.01).on(0)
ansatz += DepolarizingChannel(0.01).on(1)

ham = Hamiltonian(QubitOperator("Z0 Z1") + QubitOperator("X0", 0.5))
sim = Simulator("mqvector", 2)
grad_ops = sim.get_expectation_with_grad(ham, ansatz)


def cost(params):
    f, _ = grad_ops(params)
    return np.real(f)[0, 0]


# Use Nelder-Mead for noisy landscapes (gradient-free)
result = minimize(cost, np.zeros(2), method="Nelder-Mead")
print(f"Noisy VQE energy: {result.fun:.6f}")
```

## Performance and Feasibility Guide

| Qubits | mqvector (Monte Carlo) | mqmatrix (Density Matrix) |
|--------|----------------------|--------------------------|
| 4 | ✅ instant | ✅ instant |
| 10 | ✅ instant | ✅ ~16 MB |
| 13 | ✅ fast | ⚠️ ~1 GB |
| 15 | ✅ fast | ⚠️ ~16 GB raw state |
| 20 | ✅ fast | ❌ ~16 TB |
| 25 | ✅ moderate | ❌ impossible |
| 30 | ⚠️ ~16 GB RAM | ❌ impossible |

**For large noisy simulations (>12 qubits):** Use `mqvector` with noise channels and Monte Carlo sampling. Increase `shots` for better statistics.
