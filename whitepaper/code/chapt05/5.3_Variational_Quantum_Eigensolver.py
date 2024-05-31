from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.algorithm.nisq import (
    get_qubit_hamiltonian,
    uccsd_singlet_generator,
    uccsd_singlet_get_packed_amplitudes,
)
from mindquantum import X, Circuit, Transform, TimeEvolution, Simulator, Hamiltonian
from scipy.optimize import minimize
import numpy as np

# Step 1: Define the LiH molecular geometry
dist = 1.5
geometry = [["Li", [0.0, 0.0, 0.0 * dist]], ["H", [0.0, 0.0, 1.0 * dist]]]
basis = "sto3g"
spin = 0
molecule = MolecularData(geometry, basis, multiplicity=2 * spin + 1)
molecule = run_pyscf(molecule, run_scf=1, run_ccsd=1, run_fci=1)

# Step 2: Build the Hamiltonian for the LiH molecule
hamiltonian_QubitOP = get_qubit_hamiltonian(molecule)

# Step 3: Choose a wavefunction ansatz (UCCSD in this case)
ucc_fermion_ops = uccsd_singlet_generator(
    molecule.n_qubits, molecule.n_electrons, anti_hermitian=True
)
ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
ansatz_parameter_names = ansatz_circuit.params_name

# Step 4: Initialize the state using the Hartree-Fock state
hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])

# Step 5: Combine the initial state and the ansatz circuit
total_circuit = hartreefock_wfn_circuit + ansatz_circuit

# Step 6: Obtain the gradient operator for VQE
grad_ops = Simulator("mqvector", total_circuit.n_qubits).get_expectation_with_grad(
    Hamiltonian(hamiltonian_QubitOP.real), total_circuit
)

# Step 7: Get the initial parameters for the UCCSD ansatz from CCSD values
init_amplitudes_ccsd = uccsd_singlet_get_packed_amplitudes(
    molecule.ccsd_single_amps,
    molecule.ccsd_double_amps,
    molecule.n_qubits,
    molecule.n_electrons,
)
init_amplitudes_ccsd = [
    init_amplitudes_ccsd[param_i] for param_i in ansatz_parameter_names
]


# Define the cost function for the optimizer
def fun(theta, grad_ops, energy_list):
    energy, grad = grad_ops(theta)
    energy_list.append(energy)
    return np.squeeze(energy.real), np.squeeze(grad.real)


# Step 8: Minimize the cost function using a classical optimizer (SciPy)
energy_list = []
res = minimize(
    fun, init_amplitudes_ccsd, args=(grad_ops, energy_list), method="bfgs", jac=True
)

# Output the result
for step, energy in enumerate(energy_list):
    print(f"Step: {step}, energy: {energy}")
