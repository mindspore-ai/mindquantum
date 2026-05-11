# Built-in Ansatz Catalog

MindQuantum provides ready-to-use ansatz circuits in `mindquantum.algorithm.nisq`. All ansätze inherit from `Ansatz` base class and produce a `Circuit` via `.circuit`.

## Hardware Efficient Ansatz (HEA)

General-purpose ansatz with alternating rotation and entangling layers.

```python
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz
from mindquantum.core.gates import RY, RZ, X

ansatz = HardwareEfficientAnsatz(
    n_qubits=4,
    single_rot_gate_seq=[RY, RZ],  # rotation gates per layer
    entangle_gate=X,  # X becomes a controlled-X on entangling pairs
    depth=3,  # number of layers
)
circ = ansatz.circuit
```

Constructs alternating layers of single-qubit rotation gates and entangling gates according to `entangle_mapping`.

## Strongly Entangling Ansatz

Rotation layers with entangling gates at varying distances.

```python
from mindquantum.algorithm.nisq import StronglyEntangling
from mindquantum.core.gates import X

ansatz = StronglyEntangling(n_qubits=4, depth=1, entangle_gate=X)
circ = ansatz.circuit
```

Constructs layers of `U3` rotations followed by entangling gates at layer-dependent offsets.

## QAOA Ansatz

Builds a QAOA circuit from a `QubitOperator` cost Hamiltonian:

```python
from mindquantum.algorithm.nisq import QAOAAnsatz
from mindquantum.core.operators import QubitOperator

# Cost Hamiltonian from graph edges
ham = QubitOperator("Z0 Z1") + QubitOperator("Z1 Z2") + QubitOperator("Z0 Z2")

ansatz = QAOAAnsatz(ham, depth=4)
circ = ansatz.circuit  # includes initial H layer + p layers of cost/mixer
```

Use `QAOAAnsatz` after the target problem has already been encoded as a `QubitOperator` Hamiltonian.

## Max-Cut Ansatz

Specialized QAOA variant for Max-Cut problems:

```python
from mindquantum.algorithm.nisq import MaxCutAnsatz

ansatz = MaxCutAnsatz(graph=[(0, 1), (1, 2), (2, 0)], depth=4)  # edge list
circ = ansatz.circuit
```

## UCC Ansatz (Quantum Chemistry)

Unitary Coupled-Cluster Singles and Doubles for molecular ground state:

```python
from mindquantum.algorithm.nisq import UCCAnsatz

ansatz = UCCAnsatz(
    n_qubits=4, n_electrons=2, occ_orb=None, vir_orb=None, trotter_step=1  # occupied orbitals  # virtual orbitals
)
circ = ansatz.circuit
```

The source note states that `UCCAnsatz` uses the Jordan-Wigner transform and does not include the Hartree-Fock reference state.

For molecule-driven UCCSD with CCSD amplitudes and Hamiltonian generation, use `generate_uccsd(mol)` from the
`mq-quantum-chemistry` skill.

## Hardware Efficient for Chemistry

Hardware-efficient circuit with a separately prepared Hartree-Fock-style initial state:

```python
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz

# Use chemistry-tailored initial state + HEA
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY, RZ, X

init_state = Circuit()
init_state += X.on(0)  # |01⟩ HF reference for 2 electrons
init_state += X.on(1)

ansatz = HardwareEfficientAnsatz(4, [RY, RZ], X, depth=2)
full = init_state + ansatz.circuit
```

## IQP Encoding

Instantaneous Quantum Polynomial encoding for data re-uploading:

```python
from mindquantum.algorithm.nisq import IQPEncoding

encoding = IQPEncoding(n_feature=4, num_repeats=2)
circ = encoding.circuit
```

`IQPEncoding.data_preparation(data)` returns the parameter values expected by the generated encoding circuit.

## Ansatz Map

| Class / helper | Input it expects | Output / behavior |
|----------------|------------------|-------------------|
| `HardwareEfficientAnsatz` | `n_qubits`, rotation gates, entangling gate, mapping, depth | Alternating rotation and entangling layers |
| `StronglyEntangling` | `n_qubits`, depth, non-parameterized entangling gate | `U3` rotation layers plus entangling layers |
| `QAOAAnsatz` | `QubitOperator` Hamiltonian and depth | Initial Hadamards plus QAOA cost/mixer layers |
| `MaxCutAnsatz` | Max-Cut edge list and depth | Max-Cut-specific QAOA ansatz |
| `UCCAnsatz` | Qubit/electron counts and optional occupied/virtual orbital lists | UCC circuit using Jordan-Wigner transform; no reference state included |
| `generate_uccsd` | OpenFermion `MolecularData` or molecule filename | UCCSD circuit, initial amplitudes, parameter names, qubit Hamiltonian, qubit/electron counts |
| `IQPEncoding` | Feature count, rotation gates, repeat count | Encoding circuit plus `data_preparation()` helper |

## Key Properties

All ansätze expose:

```python
ansatz.circuit  # The parameterized Circuit
ansatz.n_qubits  # Number of qubits
ansatz.circuit.params_name  # List of circuit parameter names
```
