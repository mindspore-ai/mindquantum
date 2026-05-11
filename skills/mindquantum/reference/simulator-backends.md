# Simulator Backends Reference

## Simulator Construction

```python
from mindquantum.simulator import Simulator, get_supported_simulator
import mindquantum as mq

# Check available backends
print(get_supported_simulator())

# Create simulator
sim = Simulator("mqvector", n_qubits=4)
sim = Simulator("mqvector", 4, seed=42)
sim = Simulator("mqvector", 4, dtype=mq.complex64)
```

## Backend Comparison

| Backend | State | Memory | Availability / feature notes |
|---------|-------|--------|------------------------------|
| `mqvector` | Pure state vector | O(2^n) | CPU state-vector backend; supports expectation and gradient APIs |
| `mqvector_gpu` | Pure state vector | O(2^n) | Available only when CUDA GPU support is built and detected |
| `mqvector_cq` | Pure state vector | O(2^n) | Available only when CUDA and cuQuantum support are built and detected |
| `mqmatrix` | Density matrix | O(4^n) | Native mixed-state operations; `get_expectation_with_grad` rejects `circ_left` and `simulator_left` |
| `mqmps` | MPS tensor train | O(n chi^2) | Bond-dimension settings control exact/approximate behavior |
| `stabilizer` | Clifford tableau | O(n^2) | Clifford tableau backend; does not support parameterized gates or gradient computation |

### Memory Requirements (mqvector, complex128)

| Qubits | Memory |
|--------|--------|
| 16 | 1 MB |
| 20 | 16 MB |
| 25 | 512 MB |
| 26 | 1 GB |
| 30 | 16 GB |
| 36 | 1 TB |

## State Operations

### Apply Gates and Circuits

```python
sim.apply_gate(H.on(0))  # Single gate
sim.apply_circuit(circ)  # Non-parameterized circuit
sim.apply_circuit(circ, pr={"theta": 0.5})  # Parameterized circuit
```

### Get Quantum State

```python
sim.get_qs()  # numpy array of amplitudes
sim.get_qs(ket=True)  # Human-readable ket notation
```

### Reset

```python
sim.reset()  # Reset to |0...0⟩
sim.set_qs(state_vector)  # Set to arbitrary state
```

## Sampling

Performs measurement without modifying simulator state:

```python
# Circuit must contain Measure gates
measured_circ = circ + Measure().on(0) + Measure().on(1)

result = sim.sampling(measured_circ, shots=1000)
result = sim.sampling(measured_circ, pr={"a": 0.5}, shots=1000)

# Access results
result.data  # Dict: {'00': 503, '11': 497}
result.svg()  # Histogram visualization
```

## Expectation Values

### Basic Expectation

```python
ham = Hamiltonian(QubitOperator("Z0 Z1"))
sim.reset()
sim.apply_circuit(circ, pr={"theta": 0.5})
exp = sim.get_expectation(ham)  # <ψ|H|ψ> as complex number
```

### Expectation with Gradient

This is the most important API for variational algorithms:

```python
sim = Simulator("mqvector", n_qubits)
grad_ops = sim.get_expectation_with_grad(
    hams,  # Hamiltonian or list of Hamiltonians
    circ_right,  # The parameterized circuit U(θ)
    circ_left=None,  # Optional left circuit (for inner products)
    simulator_left=None,  # Optional left state (for ⟨φ|U†HU|ψ⟩)
    parallel_worker=None,  # Parallel threads for multiple Hamiltonians
)

# Call the grad_ops
import numpy as np

encoder_data = np.array([[0.1, 0.2]])  # [batch_size, n_encoder_params]
ansatz_data = np.array([0.3, 0.4])  # [n_ansatz_params]

f, g_enc, g_ans = grad_ops(encoder_data, ansatz_data)

# Output shapes:
# f:     [batch_size, n_hams]                    — expectation values
# g_enc: [batch_size, n_hams, n_encoder_params]  — encoder gradients
# g_ans: [batch_size, n_hams, n_ansatz_params]   — ansatz gradients
```

### For Ansatz-Only Circuits

When there are no encoder parameters:

```python
ansatz = Circuit().ry("w0", 0).ry("w1", 1).x(1, 0)
grad_ops = sim.get_expectation_with_grad(ham, ansatz)

# Only pass ansatz data
f, g = grad_ops(np.array([0.1, 0.2]))
# f shape: [1, n_hams]
# g shape: [1, n_hams, n_ansatz_params]
```

### Multiple Hamiltonians

Compute expectations and gradients for several observables at once:

```python
hams = [Hamiltonian(QubitOperator("Z0")), Hamiltonian(QubitOperator("X0")), Hamiltonian(QubitOperator("Z0 Z1"))]
grad_ops = sim.get_expectation_with_grad(hams, circ, parallel_worker=4)
# f shape: [batch_size, 3]
```

### Inner Products

Calculate ⟨φ|ψ(θ)⟩ and its gradient:

```python
sim_left = Simulator("mqvector", n_qubits)
sim_left.apply_circuit(target_state_circuit)

ham_identity = Hamiltonian(QubitOperator(""))
grad_ops = sim.get_expectation_with_grad(ham_identity, circ_right, Circuit(), simulator_left=sim_left)
```

## Density Matrix Operations (mqmatrix only)

```python
sim = Simulator("mqmatrix", 2)
sim.apply_circuit(noisy_circ)

sim.get_partial_trace([0])  # Trace out qubit 0
sim.entropy()  # Von Neumann entropy
sim.purity()  # Purity of the state
# psi = sim.get_pure_state_vector()  # Only valid if the density matrix is pure
```

## Usage Notes

1. **Check installed backends:** `get_supported_simulator()` returns the backends available in the current environment; do not assume GPU backends are present.

2. **Reuse simulators:** `sim.reset()` returns the simulator to the zero state without constructing a new `Simulator`.

3. **Use `parallel_worker`:** When computing multiple Hamiltonian expectations, set `parallel_worker` to use parallel workers where supported.

4. **Precision:** `Simulator(..., dtype=...)` supports MindQuantum dtypes such as `mindquantum.complex64` and `mindquantum.complex128`; the default is `complex128`.

5. **Batch encoder data:** Pass multiple data points as rows in a 2D array to `grad_ops()`.
