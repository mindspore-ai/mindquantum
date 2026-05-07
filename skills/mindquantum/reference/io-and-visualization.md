# I/O and Visualization Reference

## Circuit Visualization

### SVG Rendering (Jupyter)

```python
circ = Circuit().h(0).x(1, 0).rx("a", 0).measure_all()
circ.svg()  # Render in Jupyter notebook
circ.svg("dark")  # Dark mode
circ.svg("light")  # Light mode
circ.svg().to_file("circuit.svg")  # Save to file
```

### Text Representation

```python
print(circ)
# Output:
#       ┏━━━┓ ┏━━━━━━━━┓ ┏━━━┓
# q0: ──┨ H ┠─┨        ┠─┨ RX ┠─┨ M ┠──
#       ┗━━━┛ ┃        ┃ ┗━━━━┛ ┗━━━┛
#             ┃ CNOT   ┃        ┏━━━┓
# q1: ───────┨        ┠────────┨ M ┠──
#             ┗━━━━━━━━┛        ┗━━━┛
```

### Measurement Result Visualization

```python
result = sim.sampling(circ, shots=1000)
result.svg()  # Bar chart histogram
```

## OpenQASM Support

### Export to OpenQASM

```python
from mindquantum.io.qasm import OpenQASM

# Circuit → OpenQASM string
openqasm = OpenQASM()
qasm_str = openqasm.to_string(circ)
print(qasm_str)

# Save to file
openqasm.to_file(circ, "circuit.qasm")
```

### Import from OpenQASM

```python
# OpenQASM string → Circuit
qasm_str = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""
circ = OpenQASM().from_string(qasm_str)

# From file
circ = OpenQASM().from_file("circuit.qasm")
```

### Supported QASM Gates

Standard OpenQASM 2.0 gates are supported: `h`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg`, `cx`, `cz`, `ccx`, `rx`, `ry`, `rz`, `u1`, `u2`, `u3`, `swap`.

## HiQASM Support

Huawei's native quantum assembly format:

```python
from mindquantum.io.qasm import HiQASM

hiqasm = HiQASM()
hiqasm_str = hiqasm.to_string(circ)
circ = hiqasm.from_string(hiqasm_str)
```

## QCIS Support

Quantum Circuit Instruction Set (used by some hardware):

```python
from mindquantum.io.qasm import QCIS

qcis = QCIS()
qcis_str = qcis.to_string(circ)
circ = qcis.from_string(qcis_str)
```

## Hardware Topology Visualization

```python
from mindquantum.device import GridQubits, LinearQubits
from mindquantum.io.display import draw_topology

topology = GridQubits(3, 3)  # 3×3 grid
draw_topology(topology)  # Visualize topology

# After compilation, show used edges
draw_topology(topology, compiled_circuit)
```

## Circuit Summary

```python
circ.summary()
# Output:
# ╭──────────────────────╮
# │ Circuit Summary      │
# ├──────────────────────┤
# │ Qubits     : 3       │
# │ Gates      : 8       │
# │ Non-params : 4       │
# │ Params     : 4       │
# │ Encoder    : x0, x1  │
# │ Ansatz     : w0, w1  │
# ╰──────────────────────╯
```
