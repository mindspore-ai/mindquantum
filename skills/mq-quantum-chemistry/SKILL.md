---
name: mq-quantum-chemistry
description: "Run quantum chemistry simulations with MindQuantum. Covers the molecule-to-VQE pipeline: molecular definition, Hartree-Fock reference states, FermionOperator construction, fermion-to-qubit transforms (Jordan-Wigner, Parity, Bravyi-Kitaev, ternary tree, Bravyi-Kitaev Superfast), UCCSD and HEA ansätze, VQE optimization, and the mqchem CI-subspace module. Use when the user wants to simulate molecules, compute ground state energies, do quantum chemistry, use UCCSD, run VQE for chemistry, map fermion operators to qubits, or use mqchem."
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
    run_scf=True,  # Hartree-Fock
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

# Jordan-Wigner transform
qop_jw = Transform(fop).jordan_wigner()

# Parity transform
qop_p = Transform(fop).parity()

# Bravyi-Kitaev transform
qop_bk = Transform(fop).bravyi_kitaev()

# Ternary-tree transform
qop_tt = Transform(fop).ternary_tree()

# Bravyi-Kitaev Superfast transform
qop_bks = Transform(fop).bravyi_kitaev_superfast()
```

### Step 4: Ansatz Construction

#### UCCSD

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

#### Hardware-Efficient Ansatz

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


# Use any SciPy optimizer compatible with this value-and-gradient function.
result = minimize(energy_and_grad, init_amplitudes, method="L-BFGS-B", jac=True, options={"maxiter": 500})

print(f"VQE energy: {result.fun:.8f} Ha")
print(f"Difference from FCI: {abs(result.fun - mol.fci_energy):.8f} Ha")
```

## mqchem: CI-Subspace Chemistry

MindQuantum's `mqchem` module operates in a Configuration Interaction subspace instead of the full Hilbert space.

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

## Source-Backed Notes

1. **Default simulator precision:** `Simulator(..., dtype=None)` uses `mindquantum.complex128`; pass `dtype` explicitly if a different precision is required.
2. **`generate_uccsd`:** This helper returns the UCCSD circuit, initial amplitudes, parameter names, qubit Hamiltonian, qubit count, and electron count.
3. **Reference states:** `UCCAnsatz` does not include the Hartree-Fock reference state; prepare it separately before appending the ansatz circuit.
4. **Reference energies:** If `run_pyscf(..., run_fci=True)` was used and `mol.fci_energy` is available, compare VQE output to that reference explicitly.
5. **Dependencies:** Quantum chemistry workflows using `MolecularData` and `run_pyscf` require `openfermion` and `openfermionpyscf`.
