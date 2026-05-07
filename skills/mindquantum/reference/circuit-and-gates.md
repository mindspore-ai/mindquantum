# Circuit and Gates Reference

## Circuit Construction

```python
from mindquantum.core.circuit import Circuit, UN, apply, dagger
from mindquantum.core.gates import H, X, Y, Z, RX, RY, RZ, CNOT, SWAP, Measure
```

### Creating Circuits

```python
# Method 1: Incremental construction
circ = Circuit()
circ += H.on(0)
circ += X.on(1, 0)  # CNOT: X on qubit 1 controlled by qubit 0

# Method 2: Chained method calls
circ = Circuit().h(0).rx("a", 0).ry("b", 1).x(1, 0)

# Method 3: From gate list
circ = Circuit([H.on(0), X.on(1, 0), RZ("theta").on(1)])
```

### Circuit Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Concatenate | `circ1 + circ2` | Append circ2 after circ1 |
| Repeat | `circ * n` | Repeat circuit n times |
| Inverse | `circ.hermitian()` or `dagger(circ)` | Adjoint of the circuit |
| Apply to qubits | `apply(circ, [2, 3])` | Remap circuit onto different qubits |
| Barrier | `circ.barrier()` | Insert visual barrier (no physical effect) |
| Summary | `circ.summary()` | Print gate count, qubit count, parameters |
| N qubits | `circ.n_qubits` | Number of qubits used |
| Parameters | `circ.params_name` | List of parameter names |

### Universal Application (`UN`)

Apply a gate uniformly to multiple qubits:

```python
UN(H, range(5))  # H on qubits 0,1,2,3,4
UN(Measure(), range(5))  # Measure all 5 qubits
UN(X, [0, 2, 4])  # X on qubits 0, 2, 4
```

### Measurement

```python
circ += Measure().on(0)  # Measure qubit 0
circ += Measure("label").on(1)  # Measure with label
circ.measure_all()  # Adds measurements in place and returns circ
```

### Matrix and State

```python
circ.matrix()  # Unitary matrix (non-parameterized)
circ.matrix({"theta": 0.5})  # Unitary with parameter values
circ.get_qs(pr={"a": 0.1}, ket=True)  # State vector output
```

## Gate Catalog

### Non-Parameterized Gates

| Gate | Usage | Matrix |
|------|-------|--------|
| `I` | `I.on(0)` | Identity |
| `H` | `H.on(0)` | Hadamard |
| `X` | `X.on(0)` | Pauli-X (NOT) |
| `Y` | `Y.on(0)` | Pauli-Y |
| `Z` | `Z.on(0)` | Pauli-Z |
| `S` | `S.on(0)` | Phase gate (√Z) |
| `T` | `T.on(0)` | π/8 gate (√S) |
| `SX` | `SX.on(0)` | √X |
| `SWAP` | `SWAP.on([0, 1])` | Swap two qubits |
| `ISWAP` | `ISWAP.on([0, 1])` | iSWAP |

### Controlled Gates

Any gate can be controlled:

```python
X.on(1, 0)  # CNOT (also available as CNOT.on(1, 0))
X.on(0, [1, 2])  # Toffoli (CCX)
Z.on(1, 0)  # CZ
H.on(1, 0)  # Controlled-H
SWAP.on([0, 1], 2)  # Controlled-SWAP (Fredkin)
```

### Single-Qubit Rotation Gates

| Gate | Usage | Rotation Axis |
|------|-------|--------------|
| `RX` | `RX('θ').on(0)` | X-axis |
| `RY` | `RY('θ').on(0)` | Y-axis |
| `RZ` | `RZ('θ').on(0)` | Z-axis |
| `PhaseShift` | `PhaseShift('φ').on(0)` | Phase |
| `GlobalPhase` | `GlobalPhase('φ').on(0)` | Global |
| `Rn` | `Rn('α','β','γ').on(0)` | Arbitrary axis |
| `U3` | `U3('θ','φ','λ').on(0)` | Universal single-qubit |

### Two-Qubit Rotation Gates

| Gate | Usage | Description |
|------|-------|-------------|
| `Rxx` | `Rxx('θ').on([0,1])` | XX rotation |
| `Ryy` | `Ryy('θ').on([0,1])` | YY rotation |
| `Rzz` | `Rzz('θ').on([0,1])` | ZZ rotation |
| `Rxy` | `Rxy('θ').on([0,1])` | XY rotation |
| `Rxz` | `Rxz('θ').on([0,1])` | XZ rotation |
| `Ryz` | `Ryz('θ').on([0,1])` | YZ rotation |
| `Givens` | `Givens('θ').on([0,1])` | Givens rotation |
| `SWAPalpha` | `SWAPalpha('α').on([0,1])` | Parameterized SWAP |
| `FSim` | `FSim('θ','φ').on([0,1])` | Fermionic simulation |

### Special Gates

| Gate | Usage | Description |
|------|-------|-------------|
| `Power` | `Power(H, 0.5)` | Gate raised to a numeric power |
| `UnivMathGate` | `UnivMathGate('name', matrix).on(qubits)` | Custom unitary from matrix |
| `BarrierGate` | `BarrierGate().on(qubits)` | Visual separator |
| `RotPauliString` | `RotPauliString('XYZ', 'θ').on([0,1,2])` | Rotation around Pauli string |
| `GroupedPauli` | `GroupedPauli('XYZ').on([0,1,2])` | Multi-qubit Pauli |

### Noise Channels

| Channel | Usage | Physical Process |
|---------|-------|-----------------|
| `BitFlipChannel` | `BitFlipChannel(p).on(0)` | X with probability p |
| `PhaseFlipChannel` | `PhaseFlipChannel(p).on(0)` | Z with probability p |
| `BitPhaseFlipChannel` | `BitPhaseFlipChannel(p).on(0)` | Y with probability p |
| `DepolarizingChannel` | `DepolarizingChannel(p).on(0)` | Random Pauli with probability p |
| `AmplitudeDampingChannel` | `AmplitudeDampingChannel(γ).on(0)` | Energy decay |
| `PhaseDampingChannel` | `PhaseDampingChannel(γ).on(0)` | Dephasing |
| `PauliChannel` | `PauliChannel(px, py, pz).on(0)` | Custom Pauli probabilities |
| `KrausChannel` | `KrausChannel('name', kraus_ops).on(0)` | Custom Kraus operators |
| `ThermalRelaxationChannel` | `ThermalRelaxationChannel(T1, T2, gate_time).on(0)` | Thermal relaxation |

### Parameter Expressions in Gates

```python
RX("a").on(0)  # rotation = a
RX({"a": 2}).on(0)  # rotation = 2*a
RX({"a": 2, "b": 0.5}).on(0)  # rotation = 2*a + 0.5*b
RX(1.5).on(0)  # fixed rotation = 1.5 (not parameterized)
```

## Built-in Circuit Library

```python
from mindquantum.algorithm.library import qft, general_ghz_state, general_w_state

qft(range(3))  # 3-qubit QFT circuit
qft(range(3)).hermitian()  # Inverse QFT
general_ghz_state(range(3))  # GHZ state preparation
general_w_state(range(3))  # W state preparation
```
