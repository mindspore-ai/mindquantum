"""Sample code to test SequentialUCCAnsatz feature"""
import numpy as np
from mindquantum.simulator.mqchem import SequentialUCCAnsatz
from mindquantum.core.operators import FermionOperator

# Test 1: Basic sequential UCC ansatz
print("Test 1: Basic sequential UCC ansatz")
ansatz = SequentialUCCAnsatz()
ansatz.append(FermionOperator("3^ 1", "a"))
ansatz.append(FermionOperator("4^ 2", "b"))
ansatz.append(FermionOperator("3^ 1", "c"))  # Same excitation, different parameter

print(f"Number of operators: {len(ansatz)}")
print(f"Circuit length: {len(ansatz.circuit)}")
print(f"Parameter names: {ansatz.circuit.params_name}")
print()

# Test 2: Using += operator
print("Test 2: Using += operator")
ansatz2 = SequentialUCCAnsatz()
ansatz2 += FermionOperator("2^ 0", "x")
ansatz2 += FermionOperator("3^ 1", "y")
print(f"Operators after +=: {len(ansatz2)}")
print(f"Parameters: {ansatz2.circuit.params_name}")
print()

# Test 3: Removing operators
print("Test 3: Removing operators")
ansatz3 = SequentialUCCAnsatz()
ansatz3.append(FermionOperator("0^ 1", "alpha"))
ansatz3.append(FermionOperator("2^ 3", "beta"))
ansatz3.append(FermionOperator("4^ 5", "gamma"))
print(f"Before removal: {len(ansatz3)} operators")

removed = ansatz3.remove(1)
print(f"Removed operator: {removed}")
print(f"After removal: {len(ansatz3)} operators")
print(f"Remaining parameters: {ansatz3.circuit.params_name}")
print()

# Test 4: Accessing operators
print("Test 4: Accessing operators")
ansatz4 = SequentialUCCAnsatz()
ansatz4.append(FermionOperator("1^ 0", "p1"))
ansatz4.append(FermionOperator("2^ 0", "p2"))
ansatz4.append(FermionOperator("3^ 0", "p3"))

print(f"First operator: {ansatz4[0]}")
print(f"Last operator: {ansatz4[-1]}")
print(f"Slice [0:2]: {ansatz4[0:2]}")
print()

# Test 5: With n_qubits validation
print("Test 5: With n_qubits validation")
ansatz5 = SequentialUCCAnsatz(n_qubits=6, n_electrons=2)
ansatz5.append(FermionOperator("3^ 1", "t1"))
ansatz5.append(FermionOperator("5^ 2", "t2"))
print(f"Valid operators added: {len(ansatz5)}")
try:
    ansatz5.append(FermionOperator("7^ 1", "t3"))  # Should fail - exceeds n_qubits
except ValueError as e:
    print(f"Expected error caught: {e}")
print()

# Test 6: Multi-layer structure
print("Test 6: Multi-layer structure (same excitation multiple times)")
ansatz6 = SequentialUCCAnsatz()
for i in range(4):
    ansatz6.append(FermionOperator("3^ 1", f"layer{i}"))
print(f"Multi-layer ansatz has {len(ansatz6)} layers")
print(f"Parameters: {ansatz6.circuit.params_name}")
print()

# Test 7: Circuit properties
print("Test 7: Circuit properties")
ansatz7 = SequentialUCCAnsatz()
ansatz7.append(FermionOperator("2^ 0", "theta1"))
ansatz7.append(FermionOperator("3^ 1", "theta2"))
circ = ansatz7.circuit
print(f"Circuit summary:")
print(circ.summary())
print()

# Test 8: Operators list property
print("Test 8: Operators list property")
ansatz8 = SequentialUCCAnsatz()
ansatz8.append(FermionOperator("1^ 0", "a"))
ansatz8.append(FermionOperator("2^ 1", "b"))
ops = ansatz8.operators
print(f"Operators property returns: {len(ops)} operators")
print(f"First operator from property: {ops[0]}")
print()

print("SequentialUCCAnsatz tests completed successfully!")