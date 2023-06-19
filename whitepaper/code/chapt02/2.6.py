from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator

# Initialize a vector state simulator object with two qubit
sim = Simulator('mqvector', 2)
# mqvector simulator with 2 qubits (little endian), dtype: mindquantum.complex128.
# Current quantum state:
# 1¦00⟩

# Quantum state in ket format uses "little endian", which means qubits sorted
# from the largest to the smallest, i.e. |q1 q0⟩.

# Initialize a density matrix simulator object with two qubit
dm_sim = Simulator('mqmatrix', 2)
# mqmatrix simulator with 2 qubits (little endian), dtype: mindquantum.complex128.

# Apply a gate on this simulator
sim.apply_gate(X.on(0))
# mqvector simulator with 2 qubits (little endian), dtype: mindquantum.complex128.
# Current quantum state:
# 1¦01⟩

# Reset simulator to zero state
sim.reset()
# mqvector simulator with 2 qubits (little endian), dtype: mindquantum.complex128.
# Current quantum state:
# 1¦00⟩

# Apply a circuit on this simulator
circ = Circuit([X.on(0), X.on(1)])
sim.apply_circuit(circ)
# mqvector simulator with 2 qubits (little endian), dtype: mindquantum.complex128.
# Current quantum state:
# 1¦11⟩

# Apply hamiltonian to this simulator, i.e. calculating H¦ψ⟩.
ops = QubitOperator('Z0')
ham = Hamiltonian(ops)
sim.apply_hamiltonian(ham)
# mqvector simulator with 2 qubits (little endian), dtype: mindquantum.complex128.
# Current quantum state:
# -1¦11⟩

# Get expectation of the given hamiltonian. The hamiltonian could be non hermitian.
sim.reset()
expectation = sim.get_expectation(ham, circ)
print(expectation)

# Get current quantum state of this simulator
qs = sim.get_qs()
# [ 0.+0.j  0.+0.j  0.+0.j -1.+0.j]

# Sample the measure qubit in circuit
sim.reset()
circ.measure_all()
sim.sampling(circ, shots=100)
# shots: 100
# Keys: q1 q0│0.00     0.2         0.4         0.6         0.8         1.0
# ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
#          11│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
#            │
# {'11': 100}
