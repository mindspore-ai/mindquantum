"""Sample code to test mqmps backend"""
import numpy as np
import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator

# Test 1: Basic circuit with mqmps
print("Test 1: Basic circuit with mqmps backend")
n_qubits = 3
circuit = Circuit().h(0).rx(0.123, 1).x(2, 0).ry(0.456, 2).z(1, 2).rz(0.789, 0)

ref = Simulator("mqvector", n_qubits)
dut = Simulator("mqmps", n_qubits)
ref.apply_circuit(circuit)
dut.apply_circuit(circuit)

ref_qs = ref.get_qs()
dut_qs = dut.get_qs()
print(f"mqvector state shape: {ref_qs.shape}")
print(f"mqmps state shape: {dut_qs.shape}")
print(f"States match: {np.allclose(dut_qs, ref_qs, atol=1e-6)}")
print()

# Test 2: Expectation value
print("Test 2: Expectation value")
from mindquantum.core.operators import QubitOperator, Hamiltonian
ham = Hamiltonian(QubitOperator('Z0 Z1') + QubitOperator('X0'))
exp_ref = ref.get_expectation(ham)
exp_mps = dut.get_expectation(ham)
print(f"mqvector expectation: {exp_ref}")
print(f"mqmps expectation: {exp_mps}")
print(f"Match: {np.isclose(exp_ref, exp_mps)}")
print()

# Test 3: Sampling
print("Test 3: Sampling")
circ_with_measure = Circuit().h(0).h(1).rx(0.5, 2)
circ_with_measure += mq.core.gates.Measure().on(0)
circ_with_measure += mq.core.gates.Measure().on(1)
circ_with_measure += mq.core.gates.Measure().on(2)

ref2 = Simulator("mqvector", n_qubits)
mps2 = Simulator("mqmps", n_qubits)
samples_ref = ref2.sampling(circ_with_measure, shots=1000, seed=42)
samples_mps = mps2.sampling(circ_with_measure, shots=1000, seed=42)
print(f"mqvector sampling: {samples_ref}")
print(f"mqmps sampling: {samples_mps}")
print()

# Test 4: Reduced density matrix
print("Test 4: Reduced density matrix")
rdm_ref = ref.get_reduced_density_matrix([0, 1])
rdm_mps = dut.get_reduced_density_matrix([0, 1])
print(f"mqvector RDM shape: {rdm_ref.shape}")
print(f"mqmps RDM shape: {rdm_mps.shape}")
print(f"RDMs match: {np.allclose(rdm_ref, rdm_mps, atol=1e-6)}")
print()

# Test 5: Set and get quantum state
print("Test 5: Set and get quantum state")
rng = np.random.default_rng(42)
init_state = rng.random(2**n_qubits) + 1j * rng.random(2**n_qubits)
init_state = init_state / np.linalg.norm(init_state)

ref3 = Simulator("mqvector", n_qubits)
mps3 = Simulator("mqmps", n_qubits)
ref3.set_qs(init_state)
mps3.set_qs(init_state)

print(f"Set states match: {np.allclose(ref3.get_qs(), mps3.get_qs(), atol=1e-6)}")
print()

print("mqmps tests completed successfully!")