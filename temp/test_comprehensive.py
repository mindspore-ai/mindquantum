"""Comprehensive tests to find bugs in new features"""
import numpy as np
import warnings

print("=" * 70)
print("COMPREHENSIVE BUG TESTING FOR MINDQUANTUM 0.12.0 NEW FEATURES")
print("=" * 70)

# ============================================================================
# 1. QPU Tests
# ============================================================================
print("\n" + "=" * 70)
print("1. QPU BUG TESTS")
print("=" * 70)

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

# Test 1.1: Check if fd_gap validation works
print("\nTest 1.1: fd_gap validation")
try:
    qpu = DummyQPU(n_qubits=2)
    circ = Circuit().ry('a', 0).as_ansatz()
    ham = Hamiltonian(QubitOperator('Z0'))
    # Test negative fd_gap
    grad_ops = qpu.get_expectation_with_grad(ham, circ, fd_gap=-0.001)
    print("  BUG: Negative fd_gap should raise ValueError!")
except ValueError as e:
    print(f"  OK: Negative fd_gap correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 1.2: Check if zero fd_gap validation works
print("\nTest 1.2: Zero fd_gap validation")
try:
    qpu = DummyQPU(n_qubits=2)
    circ = Circuit().ry('a', 0).as_ansatz()
    ham = Hamiltonian(QubitOperator('Z0'))
    grad_ops = qpu.get_expectation_with_grad(ham, circ, fd_gap=0)
    print("  BUG: Zero fd_gap should raise ValueError!")
except ValueError as e:
    print(f"  OK: Zero fd_gap correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 1.3: Check circuit with measurement gate
print("\nTest 1.3: Circuit with measurement gate")
try:
    qpu = DummyQPU(n_qubits=2)
    circ = Circuit().ry('a', 0).measure(0).as_ansatz()
    ham = Hamiltonian(QubitOperator('Z0'))
    grad_ops = qpu.get_expectation_with_grad(ham, circ)
    print("  BUG: Circuit with measurement should raise ValueError!")
except ValueError as e:
    print(f"  OK: Measurement gate correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 1.4: Check if n_qubits validation works
print("\nTest 1.4: n_qubits validation (circuit exceeds QPU qubits)")
try:
    qpu = DummyQPU(n_qubits=2)
    circ = Circuit().ry('a', 3).as_ansatz()  # Uses qubit 3, but QPU only has 2
    ham = Hamiltonian(QubitOperator('Z0'))
    grad_ops = qpu.get_expectation_with_grad(ham, circ)
    print("  BUG: Circuit exceeding QPU qubits should raise ValueError!")
except ValueError as e:
    print(f"  OK: Circuit exceeding QPU qubits correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# ============================================================================
# 2. mqmps Tests
# ============================================================================
print("\n" + "=" * 70)
print("2. mqmps BUG TESTS")
print("=" * 70)

# Test 2.1: Large qubit system warning
print("\nTest 2.1: Large qubit system (checking for memory issues)")
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Create a moderately large system
        n_qubits = 15
        mps_sim = Simulator("mqmps", n_qubits)
        circ = Circuit()
        for i in range(n_qubits):
            circ += Circuit().h(i)
        mps_sim.apply_circuit(circ)
        qs = mps_sim.get_qs()
        print(f"  OK: Created {n_qubits}-qubit MPS simulator, state shape: {qs.shape}")
        if len(w) > 0:
            for warning in w:
                print(f"  Warning: {warning.message}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 2.2: Non-adjacent qubit operations
print("\nTest 2.2: Non-adjacent qubit two-qubit gates")
try:
    from mindquantum.core.gates import X
    mps_sim = Simulator("mqmps", 5)
    circ = Circuit().x(0) + X.on(4, 0)  # Non-adjacent CNOT using X.on(target, control)
    mps_sim.apply_circuit(circ)
    print(f"  OK: Non-adjacent CNOT works")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 2.3: get_reduced_density_matrix edge cases
print("\nTest 2.3: get_reduced_density_matrix edge cases")
try:
    from mindquantum.core.gates import X
    mps_sim = Simulator("mqmps", 3)
    circ = Circuit().h(0) + X.on(1, 0)  # CNOT using X.on(target, control)
    mps_sim.apply_circuit(circ)
    
    # Test with single qubit
    rdm1 = mps_sim.get_reduced_density_matrix([0])
    print(f"  Single qubit RDM shape: {rdm1.shape}")
    
    # Test with all qubits (should fail)
    try:
        rdm_all = mps_sim.get_reduced_density_matrix([0, 1, 2])
        print("  BUG: get_reduced_density_matrix with all qubits should raise ValueError!")
    except ValueError as e:
        print(f"  OK: All qubits correctly rejected: {e}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# ============================================================================
# 3. SequentialUCCAnsatz Tests
# ============================================================================
print("\n" + "=" * 70)
print("3. SequentialUCCAnsatz BUG TESTS")
print("=" * 70)

from mindquantum.simulator.mqchem import SequentialUCCAnsatz
from mindquantum.core.operators import FermionOperator

# Test 3.1: Empty ansatz
print("\nTest 3.1: Empty ansatz")
try:
    ansatz = SequentialUCCAnsatz()
    circ = ansatz.circuit
    print(f"  OK: Empty ansatz circuit has {len(circ)} gates")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 3.2: Identity operator (should fail)
print("\nTest 3.2: Identity operator (should fail)")
try:
    ansatz = SequentialUCCAnsatz()
    ansatz.append(FermionOperator("", 1.0))  # Identity
    print("  BUG: Identity operator should raise ValueError!")
except ValueError as e:
    print(f"  OK: Identity operator correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 3.3: Multi-term operator (should fail)
print("\nTest 3.3: Multi-term operator (should fail)")
try:
    ansatz = SequentialUCCAnsatz()
    ansatz.append(FermionOperator("1^ 0", "a") + FermionOperator("2^ 1", "b"))
    print("  BUG: Multi-term operator should raise ValueError!")
except ValueError as e:
    print(f"  OK: Multi-term operator correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 3.4: Invalid index access
print("\nTest 3.4: Invalid index access")
try:
    ansatz = SequentialUCCAnsatz()
    ansatz.append(FermionOperator("1^ 0", "a"))
    op = ansatz[10]  # Out of range
    print("  BUG: Out of range index should raise IndexError!")
except IndexError as e:
    print(f"  OK: Out of range index correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 3.5: Remove from empty ansatz
print("\nTest 3.5: Remove from empty ansatz")
try:
    ansatz = SequentialUCCAnsatz()
    ansatz.remove(0)
    print("  BUG: Remove from empty ansatz should raise IndexError!")
except IndexError as e:
    print(f"  OK: Remove from empty ansatz correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 3.6: Negative n_qubits
print("\nTest 3.6: Negative n_qubits")
try:
    ansatz = SequentialUCCAnsatz(n_qubits=-1)
    print("  BUG: Negative n_qubits should raise ValueError!")
except ValueError as e:
    print(f"  OK: Negative n_qubits correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# ============================================================================
# 4. QAIA Tests
# ============================================================================
print("\n" + "=" * 70)
print("4. QAIA (TSB, USB, LSB) BUG TESTS")
print("=" * 70)

from mindquantum.algorithm.qaia import TSB, USB, LSB
from scipy.sparse import csr_matrix

# Create test problem
np.random.seed(42)
N = 5
J_dense = np.random.randn(N, N)
J_dense = (J_dense + J_dense.T) / 2
np.fill_diagonal(J_dense, 0)
J = csr_matrix(J_dense)

# Test 4.1: Invalid strategy for TSB
print("\nTest 4.1: Invalid strategy for TSB")
try:
    tsb = TSB(J, strategy="invalid_strategy")
    tsb.update()
    print("  BUG: Invalid strategy should raise error!")
except (ValueError, KeyError) as e:
    print(f"  OK: Invalid strategy correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 4.2: Negative threshold for USB
print("\nTest 4.2: Negative threshold for USB")
try:
    usb = USB(J, threshold=-5)
    usb.update()
    print("  INFO: Negative threshold accepted (may cause unexpected behavior)")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 4.3: Zero threshold for LSB
print("\nTest 4.3: Zero threshold for LSB")
try:
    lsb = LSB(J, threshold=0)
    lsb.update()
    print("  INFO: Zero threshold accepted (may cause unexpected behavior)")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 4.4: Non-symmetric J matrix
print("\nTest 4.4: Non-symmetric J matrix")
try:
    J_bad = np.array([[0, 1, 2], [0, 0, 3], [0, 0, 0]])  # Not symmetric
    tsb = TSB(J_bad)
    print("  BUG: Non-symmetric J should raise ValueError!")
except ValueError as e:
    print(f"  OK: Non-symmetric J correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 4.5: J with non-zero diagonal
print("\nTest 4.5: J with non-zero diagonal")
try:
    J_diag = np.array([[1, -1], [-1, 2]])  # Non-zero diagonal
    tsb = TSB(J_diag)
    print("  BUG: J with non-zero diagonal should raise ValueError!")
except ValueError as e:
    print(f"  OK: J with non-zero diagonal correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 4.6: calc_cut return type consistency
print("\nTest 4.6: calc_cut return type consistency")
try:
    # Single batch
    tsb_single = TSB(J, batch_size=1, n_iter=10)
    tsb_single.update()
    cut_single = tsb_single.calc_cut()
    print(f"  Single batch calc_cut type: {type(cut_single)}, shape: {cut_single.shape if hasattr(cut_single, 'shape') else 'N/A'}")
    
    # Multiple batch
    tsb_multi = TSB(J, batch_size=3, n_iter=10)
    tsb_multi.update()
    cut_multi = tsb_multi.calc_cut()
    print(f"  Multi batch calc_cut type: {type(cut_multi)}, shape: {cut_multi.shape if hasattr(cut_multi, 'shape') else 'N/A'}")
    
    # Check consistency
    if hasattr(cut_single, 'shape') and hasattr(cut_multi, 'shape'):
        if cut_single.shape == cut_multi.shape:
            print("  INFO: Same shape for different batch sizes")
        else:
            print("  OK: Different shapes for different batch sizes")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 4.7: Invalid backend
print("\nTest 4.7: Invalid backend")
try:
    tsb = TSB(J, backend="invalid_backend")
    print("  BUG: Invalid backend should raise ValueError!")
except ValueError as e:
    print(f"  OK: Invalid backend correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 4.8: Negative n_iter
print("\nTest 4.8: Negative n_iter")
try:
    tsb = TSB(J, n_iter=-1)
    print("  BUG: Negative n_iter should raise ValueError!")
except ValueError as e:
    print(f"  OK: Negative n_iter correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

# Test 4.9: Negative batch_size
print("\nTest 4.9: Negative batch_size")
try:
    tsb = TSB(J, batch_size=-1)
    print("  BUG: Negative batch_size should raise ValueError!")
except ValueError as e:
    print(f"  OK: Negative batch_size correctly rejected: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("COMPREHENSIVE BUG TESTING COMPLETED")
print("=" * 70)
