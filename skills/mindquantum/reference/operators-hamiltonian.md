# Operators and Hamiltonian Reference

## QubitOperator

Represents operators as sums of Pauli strings acting on qubits.

```python
from mindquantum.core.operators import QubitOperator
```

### Construction

```python
# Single Pauli string
op = QubitOperator("X0")  # X on qubit 0
op = QubitOperator("X0 Y1")  # X⊗Y on qubits 0,1
op = QubitOperator("Z0 Z1", 0.5)  # 0.5 * Z⊗Z

# Linear combinations
op = QubitOperator("X0", 0.5) + QubitOperator("Z0", -0.3)  # 0.5*X0 - 0.3*Z0

# Identity
op = QubitOperator("")  # Identity operator
op = QubitOperator("", 2.0)  # 2.0 * Identity

# From string parsing
op = QubitOperator("X0 Y1", 1.0) + QubitOperator("Z0 Z1", -0.5)
```

### Operations

```python
op1 + op2  # Addition
op1 * op2  # Multiplication (tensor product / composition)
op * 2.5  # Scalar multiplication
op.hermitian()  # Hermitian conjugate
```

### Inspection

```python
op.terms  # Dict of {pauli_string: coefficient}
op.size  # Number of terms
count_qubits(op)  # Number of qubits used
op.is_singlet  # True if single term
```

## FermionOperator

Represents fermionic creation (†) and annihilation operators.

```python
from mindquantum.core.operators import FermionOperator
```

### Construction

```python
# Creation operator on mode 0
a0_dag = FermionOperator("0^")

# Annihilation operator on mode 1
a1 = FermionOperator("1")

# Number operator: a†a
n0 = FermionOperator("0^ 0")

# Two-body term
two_body = FermionOperator("0^ 1^ 3 2", coefficient=-1.0)

# Linear combination
hop = FermionOperator("0^ 1", -1) + FermionOperator("1^ 0", -1)
```

### Operations

```python
op1 + op2  # Addition
op1 * op2  # Multiplication (respects anti-commutation)
normal_ordered(op)  # Normal ordering (creation before annihilation)
hermitian_conjugated(op)  # Hermitian conjugate
commutator(op1, op2)  # [op1, op2]
```

## Fermion-to-Qubit Transforms

Convert FermionOperators to QubitOperators for simulation:

```python
from mindquantum.algorithm.nisq import Transform

fop = FermionOperator("0^ 1") + FermionOperator("1^ 0")

# Available transforms
Transform(fop).jordan_wigner()  # Preserves locality
Transform(fop).parity()  # Reduces circuit depth
Transform(fop).bravyi_kitaev()  # Balance of both
Transform(fop).ternary_tree()  # Ternary-tree transform
Transform(fop).bravyi_kitaev_superfast()  # Superfast variant

# Reverse: QubitOperator → FermionOperator
qop = QubitOperator("X0 Z1", 0.5)
Transform(qop).reversed_jordan_wigner()
```

## Hamiltonian

Wraps a `QubitOperator` for use in simulation and gradient computation:

```python
from mindquantum.core.operators import Hamiltonian

# From QubitOperator
ham = Hamiltonian(QubitOperator("Z0 Z1") + QubitOperator("X0", 0.5))

# Used in expectation/gradient
sim = Simulator("mqvector", 2)
grad_ops = sim.get_expectation_with_grad(ham, circuit)

# Multiple Hamiltonians simultaneously
hams = [Hamiltonian(QubitOperator("Z0")), Hamiltonian(QubitOperator("X0"))]
grad_ops = sim.get_expectation_with_grad(hams, circuit)
```

## TimeEvolution

Creates a parameterized circuit from a Hamiltonian via Trotterization:

```python
from mindquantum.core.operators import TimeEvolution

# e^{-iHt} circuit
ham = QubitOperator("X0 Y1", 1.0) + QubitOperator("Z0", 0.5)
circ = TimeEvolution(ham, time={"t": 1}).circuit
```

This is essential for building QAOA ansätze and simulating time dynamics.

## Utility Functions

```python
from mindquantum.core.operators import (
    commutator,  # [A, B] = AB - BA
    count_qubits,  # Number of qubits in operator
    hermitian_conjugated,  # Hermitian conjugate
    normal_ordered,  # Normal order fermion operator
    number_operator,  # Number operator for N modes
    up_index,
    down_index,  # Spin orbital indexing
)
```

## Projector

For defining projection operators:

```python
from mindquantum.core.operators import Projector

proj = Projector("01")  # |01⟩⟨01| projector
```

## QubitExcitationOperator

For qubit excitation operators used in chemistry:

```python
from mindquantum.core.operators import QubitExcitationOperator

# Single excitation
qeo = QubitExcitationOperator("0^ 1", "theta")
circ = qeo.to_qubit_operator()
```
