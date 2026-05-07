---
name: mq-circuit-compiler
description: "Compile and optimize quantum circuits for hardware execution using MindQuantum's compiler pipeline. Covers gate decomposition into native gate sets, DAG-based circuit optimization, SABRE qubit mapping for hardware topologies (grid, linear, custom), and circuit equivalence checking. Use when the user needs to compile a circuit for a specific quantum processor, map logical qubits to physical qubits, decompose gates, optimize circuit depth, define hardware topology, or check circuit equivalence."
---

# Circuit Compilation and Hardware Mapping

MindQuantum provides a full compilation pipeline: gate decomposition → DAG optimization → qubit mapping, transforming abstract quantum circuits into hardware-executable form.

## Compilation Pipeline Overview

```text
Logical Circuit → Gate Decomposition → DAG Optimization → Qubit Mapping → Physical Circuit
                  (to native gate set)   (simplify)         (SABRE)        (with SWAPs)
```

## Hardware Topology

Define the qubit connectivity of your target device:

```python
from mindquantum.device import QubitsTopology, GridQubits, LinearQubits, QubitNode

# Predefined topologies
linear = LinearQubits(5)  # 0-1-2-3-4 chain
grid = GridQubits(3, 3)  # 3×3 grid (9 qubits)

# Custom topology
topo = QubitsTopology([QubitNode(i) for i in range(5)])
topo[0] >> topo[1]  # Connect qubit 0 ↔ 1
topo[1] >> topo[2]  # Connect qubit 1 ↔ 2
topo[2] >> topo[3]
topo[3] >> topo[4]
topo[0] >> topo[3]  # Add diagonal connection

# Inspect
print(topo.edges_with_id())  # List of (qubit_a, qubit_b) pairs
print(topo.all_qubit_id())  # List of qubit IDs
```

### Modifying Topologies

```python
# Remove a qubit (e.g., defective qubit on hardware)
topo.remove_qubit_node(2)

# Isolate a qubit (break all its connections)
topo.isolate_with_near(3)
```

### Visualizing Topologies

```python
from mindquantum.io.display import draw_topology

draw_topology(grid)  # Show topology graph
draw_topology(grid, compiled_circuit)  # Highlight used edges
```

## SABRE Qubit Mapping

The SABRE algorithm maps logical qubits to physical qubits and inserts SWAP gates to satisfy connectivity constraints.

```python
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RX, X
from mindquantum.device import GridQubits
from mindquantum.algorithm.mapping import SABRE

# 1. Define logical circuit (may have non-local gates)
circ = Circuit()
circ += H.on(0)
circ += X.on(2, 0)  # CNOT: target 2, control 0; may not be connected
circ += RX("a").on(1)
circ += X.on(3, 1)
circ += X.on(0, 3)  # qubits 0 and 3 may not be connected

# 2. Define hardware topology
topo = GridQubits(2, 2)
# Grid:  0 - 1
#         |   |
#         2 - 3

# 3. Run SABRE
solver = SABRE(circ, topo)
new_circ, init_mapping, final_mapping = solver.solve(
    iter_num=5,  # SABRE iterations (more = potentially better)
    w=0.5,  # Weight for lookahead heuristic
    delta1=0.3,  # Decay parameter for single-qubit gates
    delta2=0.2,  # Decay parameter for two-qubit gates
)

# 4. Results
print(f"Original gates: {len(circ)}")
print(f"Compiled gates: {len(new_circ)}")  # includes inserted SWAPs
print(f"Initial mapping: {init_mapping}")  # logical → physical
print(f"Final mapping: {final_mapping}")

# 5. View compiled circuit
new_circ.svg()
```

### SABRE Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `iter_num` | int | Number of SABRE iterations. Higher → better mapping, slower. Default 5. |
| `w` | float | Weight for front-layer vs lookahead cost. Range [0, 1]. |
| `delta1` | float | Decay parameter for single-qubit gates. |
| `delta2` | float | Decay parameter for two-qubit gates. |

## Gate Decomposition

Decompose complex gates into a native gate set:

```python
from mindquantum.algorithm.compiler import decompose
```

### Decomposition Rules

MindQuantum can decompose gates into standard universal gate sets. The compiler provides built-in rules for:

- Multi-controlled gates → cascaded Toffoli → CX + single-qubit
- Arbitrary unitary → U3 + CX decomposition
- Named gates (SWAP, Toffoli, etc.) → native primitives

### Using the DAG Representation

The compiler converts circuits to Directed Acyclic Graphs for optimization:

```python
from mindquantum.algorithm.compiler import DAGCircuit

# Convert circuit to DAG
dag = DAGCircuit(circ)

# DAG enables:
# - Gate cancellation (adjacent inverse gates)
# - Gate commutation (reorder independent gates)
# - Template matching (replace subcircuits)
```

## Circuit Equivalence Checking

Verify that compilation preserved circuit semantics:

### Numerical Verification

```python
import numpy as np
from mindquantum.core.circuit import Circuit, dagger

# Method 1: Matrix comparison (small circuits)
original = Circuit().h(0).x(1, 0).rx("a", 0)
compiled = Circuit().h(0).x(1, 0).rx("a", 0)  # Replace with your compiled circuit

# For fixed parameters
params = {"a": 0.5}
m1 = original.matrix(params)
m2 = compiled.matrix(params)
assert np.allclose(m1, m2), "Circuits are not equivalent!"

# Method 2: Identity check
# If A† · B = I, then A ≡ B
check = dagger(original) + compiled
m_check = check.matrix(params)
assert np.allclose(m_check, np.eye(m_check.shape[0])), "Not equivalent!"
```

### Random Parameter Verification

For parameterized circuits, test with multiple random parameter sets:

```python
param_names = original.params_name
for _ in range(10):
    pr = {name: np.random.uniform(-np.pi, np.pi) for name in param_names}
    m1 = original.matrix(pr)
    m2 = compiled.matrix(pr)
    assert np.allclose(m1, m2, atol=1e-10), f"Mismatch at params={pr}"
```

## Complete Compilation Workflow

```python
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RY, RZ, X
from mindquantum.device import GridQubits
from mindquantum.algorithm.mapping import SABRE
from mindquantum.io.display import draw_topology

# 1. Build your algorithm circuit
n_qubits = 6
circ = Circuit()
for i in range(n_qubits):
    circ += H.on(i)
for i in range(n_qubits - 1):
    circ += X.on(i + 1, i)
for i in range(n_qubits):
    circ += RY(f"theta_{i}").on(i)
# Long-range gate (not nearest-neighbor)
circ += X.on(5, 0)

# 2. Define target hardware topology
topo = GridQubits(2, 3)  # 2×3 grid for 6 qubits

# 3. Map to hardware
solver = SABRE(circ, topo)
compiled, init_map, final_map = solver.solve(10, 0.5, 0.3, 0.2)

# 4. Report
print(f"SWAPs inserted: {len(compiled) - len(circ)}")
print(f"Logical → Physical mapping: {init_map}")

# 5. Visualize
draw_topology(topo, compiled)
compiled.svg()
```

## Tips

1. **Iterate SABRE:** Run `solver.solve()` with higher `iter_num` (10-50) for better results on large circuits.
2. **Topology matters:** Choose a topology matching your target hardware. IBM devices use heavy-hex; Google uses grid.
3. **Minimize long-range gates:** SWAP insertion cost grows with qubit distance. Design circuits with local interactions when possible.
4. **Verify after compilation:** Always check equivalence between original and compiled circuits, especially for parameterized circuits.
