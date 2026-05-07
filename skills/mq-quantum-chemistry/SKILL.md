---
name: mq-quantum-chemistry
description: "Run quantum chemistry simulations with MindQuantum. Covers the molecule-to-VQE pipeline: molecular definition, Hartree-Fock reference states, FermionOperator construction, fermion-to-qubit transforms (Jordan-Wigner, Parity, Bravyi-Kitaev), UCCSD and HEA ansätze, VQE optimization, and the mqchem module for large-scale chemistry. Use when the user wants to simulate molecules, compute ground state energies, do quantum chemistry, use UCCSD, run VQE for chemistry, map fermion operators to qubits, or use mqchem for large-qubit chemistry simulations."
---

# Quantum Chemistry with MindQuantum

MindQuantum provides a complete pipeline for variational quantum chemistry, from molecular specification to ground state energy computation.

## The Chemistry Pipeline

```text
Molecule → Classical Pre-calc → FermionOperator → Qubit Transform → Ansatz → VQE → Ground State Energy
  (geometry)   (HF, integrals)    (second quant.)   (JW/Parity/BK)   (UCCSD)  (optimize)
```

## Quick Start: H₂ Ground State

```python
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.algorithm.nisq import generate_uccsd
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
import numpy as np
from scipy.optimize import minimize

# 1. Define and compute molecule classically
geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]
mol = MolecularData(geometry, "sto-3g", multiplicity=1, charge=0)
mol = run_pyscf(mol, run_ccsd=True, run_fci=True)
print(f"FCI energy: {mol.fci_energy:.6f} Ha")

# 2. Generate everything at once
ansatz_circuit, init_amplitudes, param_names, qubit_ham, n_qubits, n_electrons = generate_uccsd(mol)

# 3. Prepare Hartree-Fock initial state
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X

hf_state = Circuit()
for i in range(n_electrons):
    hf_state += X.on(i)

full_circuit = hf_state + ansatz_circuit

# 4. Run VQE
sim = Simulator("mqvector", n_qubits)
ham = Hamiltonian(qubit_ham)
grad_ops = sim.get_expectation_with_grad(ham, full_circuit)


def energy_and_grad(params):
    f, g = grad_ops(params)
    return np.real(f)[0, 0], np.real(g)[0, 0]


result = minimize(energy_and_grad, init_amplitudes, method="BFGS", jac=True)
print(f"VQE energy: {result.fun:.6f} Ha")
print(f"Error:      {abs(result.fun - mol.fci_energy):.2e} Ha")
```

## Step-by-Step Breakdown

### Step 1: Molecular Definition

```python
from openfermion import MolecularData
from openfermionpyscf import run_pyscf

# Geometry: list of (atom, (x, y, z)) in Angstroms
geometry = [("Li", (0, 0, 0)), ("H", (0, 0, 1.6))]

mol = MolecularData(geometry=geometry, basis="sto-3g", multiplicity=1, charge=0)  # Basis set  # 2S+1  # Net charge

# Run classical methods for reference energies
mol = run_pyscf(
    mol,
    run_scf=True,  # Hartree-Fock (always needed)
    run_ccsd=True,  # CCSD (provides initial amplitudes)
    run_fci=True,  # FCI (exact reference energy)
)

print(f"HF energy:   {mol.hf_energy:.6f}")
print(f"CCSD energy: {mol.ccsd_energy:.6f}")
print(f"FCI energy:  {mol.fci_energy:.6f}")
print(f"n_qubits:    {mol.n_qubits}")
print(f"n_electrons: {mol.n_electrons}")
```

### Step 2: Hamiltonian Construction

```python
from mindquantum.algorithm.nisq.chem import get_qubit_hamiltonian

# Method 1: Direct conversion
qubit_ham = get_qubit_hamiltonian(mol)

# Method 2: Manual — more control
from mindquantum.core.operators import FermionOperator, InteractionOperator

# Convert OpenFermion molecular integrals to MindQuantum FermionOperator
ham_of = mol.get_molecular_hamiltonian()
inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
fermion_ham = FermionOperator(inter_ops)

# Transform to qubit representation
from mindquantum.algorithm.nisq import Transform

qubit_ham = Transform(fermion_ham).jordan_wigner()

# Wrap for simulation
from mindquantum.core.operators import Hamiltonian

ham = Hamiltonian(qubit_ham)
```

### Step 3: Fermion-to-Qubit Transforms

MindQuantum provides these transforms:

```python
from mindquantum.algorithm.nisq import Transform

fop = fermion_hamiltonian  # FermionOperator

# Jordan-Wigner: preserves locality, simple but deep circuits
qop_jw = Transform(fop).jordan_wigner()

# Parity: stores occupation parity non-locally; tapering is a separate step
qop_p = Transform(fop).parity()

# Bravyi-Kitaev: logarithmic depth, balanced
qop_bk = Transform(fop).bravyi_kitaev()

# Ternary tree: tree-structured transform
qop_tt = Transform(fop).ternary_tree()

# Bravyi-Kitaev Superfast: for lattice Hamiltonians
qop_bks = Transform(fop).bravyi_kitaev_superfast()
```

#### Transform Selection Guide

| Transform | Circuit Depth | Qubit Count | Best For |
|-----------|--------------|-------------|----------|
| Jordan-Wigner | Deep (O(n)) | Same | Debugging, small molecules |
| Parity | Moderate | Same before any separate tapering | General use with symmetry |
| Bravyi-Kitaev | Shallow (O(log n)) | Same | Larger molecules |
| Ternary tree | Moderate | Same | Alternative tree encoding |
| Bravyi-Kitaev Superfast | Problem-dependent | Problem-dependent | Lattice-style Hamiltonians |

### Step 4: Ansatz Construction

#### UCCSD (Chemically Motivated)

```python
from mindquantum.algorithm.nisq import generate_uccsd

# All-in-one helper
circuit, init_amps, param_names, qubit_ham, n_qubits, n_elec = generate_uccsd(mol)

# Or manual construction
from mindquantum.algorithm.nisq import uccsd_singlet_generator, Transform
from mindquantum.core.operators import TimeEvolution

ucc_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons)
qubit_ucc = Transform(ucc_ops).jordan_wigner()
ansatz = TimeEvolution(qubit_ucc.imag, 1.0).circuit
```

#### Get Initial Amplitudes from CCSD

```python
from mindquantum.algorithm.nisq import uccsd_singlet_get_packed_amplitudes

init_amplitudes = uccsd_singlet_get_packed_amplitudes(
    mol.ccsd_single_amps, mol.ccsd_double_amps, mol.n_qubits, mol.n_electrons
)
```

#### Hardware-Efficient (Shallower Alternative)

```python
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz
from mindquantum.core.gates import RY, RZ, X

ansatz = HardwareEfficientAnsatz(n_qubits=mol.n_qubits, single_rot_gate_seq=[RY, RZ], entangle_gate=X, depth=4).circuit
```

### Step 5: VQE Optimization

```python
sim = Simulator("mqvector", n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, hf_state + ansatz)


def energy_and_grad(params):
    f, g = grad_ops(params)
    return np.real(f)[0, 0], np.real(g)[0, 0]


# L-BFGS-B often works best for chemistry
result = minimize(energy_and_grad, init_amplitudes, method="L-BFGS-B", jac=True, options={"maxiter": 500})

print(f"VQE energy: {result.fun:.8f} Ha")
print(f"Chemical accuracy achieved: {abs(result.fun - mol.fci_energy) < 1.6e-3}")
```

## mqchem: Large-Scale Chemistry

For molecules requiring 20+ qubits, MindQuantum's `mqchem` module operates in a Configuration Interaction subspace instead of the full Hilbert space.

```python
from mindquantum.simulator import mqchem

# 1. Prepare components from molecular data
hamiltonian, ansatz_circuit, init_amps = mqchem.prepare_uccsd_vqe(
    mol, threshold=1e-6  # Filter small excitation operators
)

# 2. Create CI-subspace simulator
vqe_sim = mqchem.MQChemSimulator(mol.n_qubits, mol.n_electrons, seed=42)

# 3. Get gradient operator
grad_ops = vqe_sim.get_expectation_with_grad(hamiltonian, ansatz_circuit)

# 4. Optimize
result = minimize(grad_ops, init_amps, method="L-BFGS-B", jac=True)
print(f"mqchem VQE energy: {result.fun:.8f} Ha")
```

### mqchem Key Classes

| Class | Purpose |
|-------|---------|
| `mqchem.CIHamiltonian` | Hamiltonian optimized for CI subspace |
| `mqchem.UCCExcitationGate` | UCC excitation as a gate: $e^{\theta(T - T^\dagger)}$ |
| `mqchem.MQChemSimulator` | Simulator operating in CI subspace |
| `mqchem.prepare_uccsd_vqe` | All-in-one: molecule → (hamiltonian, circuit, init_params) |

## Potential Energy Surface Scan

Compute energy at multiple bond lengths:

```python
import numpy as np
from scipy.optimize import minimize

distances = np.arange(0.4, 3.0, 0.1)
energies = []

for d in distances:
    geometry = [("H", (0, 0, 0)), ("H", (0, 0, d))]
    mol = MolecularData(geometry, "sto-3g", 1, 0)
    mol = run_pyscf(mol, run_ccsd=True)

    circ, init_amps, param_names, qham, nq, ne = generate_uccsd(mol)

    hf = Circuit()
    for i in range(ne):
        hf += X.on(i)

    sim = Simulator("mqvector", nq)
    grad_ops = sim.get_expectation_with_grad(Hamiltonian(qham), hf + circ)

    def cost(p):
        f, g = grad_ops(p)
        return np.real(f)[0, 0], np.real(g)[0, 0]

    res = minimize(cost, init_amps, method="BFGS", jac=True)
    energies.append(res.fun)
    print(f"d={d:.1f} Å, E={res.fun:.6f} Ha")
```

## Tips

1. **Always use `complex128`:** Chemistry calculations need double precision for chemical accuracy (~1.6 mHa).
2. **Start with CCSD amplitudes:** `init_amplitudes` from CCSD converges much faster than random initialization.
3. **Use `generate_uccsd`:** The all-in-one helper handles qubit Hamiltonian generation, UCCSD construction, and initial amplitude extraction.
4. **Check against FCI:** For small molecules, FCI provides the exact energy — your VQE should be within 1.6 mHa (chemical accuracy).
5. **mqchem for scale:** When the molecule needs >16 qubits, switch to `mqchem.MQChemSimulator` which operates in the CI subspace.
6. **Dependencies:** Quantum chemistry requires `openfermion` and `openfermionpyscf`. Install via `pip install openfermion openfermionpyscf`.
