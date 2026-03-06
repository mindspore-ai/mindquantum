"""Sample code to test QPU feature"""
import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.device import QPU
from mindquantum.simulator import Simulator


class DummyQPU(QPU):
    """A simple QPU that delegates to simulator"""
    def __init__(self, n_qubits):
        super().__init__(n_qubits, default_shots=1000)
        self.sim = Simulator('mqvector', n_qubits)
    
    def sampling(self, circuit, pr=None, shots=1, seed=None):
        return self.sim.sampling(circuit, pr=pr, shots=shots, seed=seed)


# Test 1: Basic expectation value
print("Test 1: Basic expectation value")
qpu = DummyQPU(n_qubits=2)
circ = Circuit().ry(1.2, 0)
ham = Hamiltonian(QubitOperator('Z0'))
exp_val = qpu.get_expectation(ham, circ_right=circ, shots=5000, seed=42)
print(f"Expectation value: {exp_val}")
print(f"Real part: {exp_val.real}")
print()

# Test 2: Expectation with gradient
print("Test 2: Expectation with gradient")
circ2 = Circuit().ry('a', 0).as_ansatz()
ham2 = Hamiltonian(QubitOperator('Z0'))
grad_ops = qpu.get_expectation_with_grad(ham2, circ2, pr_shift=True, shots=10000, seed=42)
f, g = grad_ops(np.array([1.0]))
print(f"Expectation shape: {f.shape}, value: {f}")
print(f"Gradient shape: {g.shape}, value: {g}")
print()

# Test 3: Multiple Hamiltonians
print("Test 3: Multiple Hamiltonians")
hams = [
    Hamiltonian(QubitOperator('Z0')),
    Hamiltonian(QubitOperator('X0'))
]
grad_ops2 = qpu.get_expectation_with_grad(hams, circ2, pr_shift=False, shots=5000, seed=42)
f2, g2 = grad_ops2(np.array([0.5]))
print(f"Multi-Ham expectation: {f2}")
print(f"Multi-Ham gradient: {g2}")
print()

# Test 4: Complex parameter expressions
print("Test 4: ParameterResolver with linear expressions")
from mindquantum.core.parameterresolver import ParameterResolver as PR
expr = 0.5 * PR('a') - 1.2 * PR('b') + 0.3
circ3 = Circuit().ry(expr, 0).as_ansatz()
grad_ops3 = qpu.get_expectation_with_grad(ham2, circ3, pr_shift=True, shots=10000, seed=42)
f3, g3 = grad_ops3(np.array([0.7, -0.2]))
print(f"Expectation with expr: {f3}")
print(f"Gradient with expr: {g3}")
print()

print("QPU tests completed successfully!")