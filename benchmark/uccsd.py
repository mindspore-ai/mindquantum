import os


os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from scipy.optimize import minimize
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import Circuit, X, Hamiltonian
from mindquantum import generate_uccsd
from mindquantum.simulator.simulator import Simulator

dist = 1.5
geometry = [
    ["Li", [0.0, 0.0, 0.0 * dist]],
    ["H", [0.0, 0.0, 1.0 * dist]],
]
basis = "sto3g"
spin = 0
print("Geometry: \n", geometry)

molecule_of = MolecularData(geometry, basis, multiplicity=2 * spin + 1)
molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=1)

molecule_of.save()
molecule_file = molecule_of.filename
print(molecule_file)

print("Hartree-Fock energy: %20.16f Ha" % (molecule_of.hf_energy))
print("CCSD energy: %20.16f Ha" % (molecule_of.ccsd_energy))
print("FCI energy: %20.16f Ha" % (molecule_of.fci_energy))

hartreefock_wfn_circuit = Circuit(
    [X.on(i) for i in range(molecule_of.n_electrons)])
print(hartreefock_wfn_circuit)

ansatz_circuit, \
init_amplitudes, \
ansatz_parameter_names, \
hamiltonian_QubitOp, \
n_qubits, n_electrons = generate_uccsd(molecule_file, th=-1)
sparsed_ham = Hamiltonian(hamiltonian_QubitOp)
sparsed_ham.sparse(n_qubits)
total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()
print("Number of parameters: %d" % (len(ansatz_parameter_names)))

molecule_pqc = Simulator('projectq', n_qubits).get_expectation_with_grad(
    sparsed_ham, total_circuit)


def energy_obj(n_paras, mol_pqc):

    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    return np.real(e[0, 0]), np.real(grad[0, 0])


n_paras = len(total_circuit.params_name)
paras = init_amplitudes

res = minimize(energy_obj,
               paras,
               args=(molecule_pqc, ),
               method='bfgs',
               jac=True,
               tol=1e-6)
energy = float(res.fun)
""" class PQCNet(ms.nn.Cell):
    def __init__(self, pqc):
        super(PQCNet, self).__init__()
        self.pqc = pqc
        self.weight =  Parameter(initializer("Zeros",
            len(self.pqc.ansatz_params_names)),
            name="weight")
        self.encoder_data_dummy = ms.Tensor([[0]], 
            self.weight.dtype)

    def construct(self):
        energy, _, grads = self.pqc(self.encoder_data_dummy, self.weight)
        return energy

molecule_pqcnet = PQCNet(molecule_pqc)

initial_energy = molecule_pqcnet()
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))

optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

eps = 1.e-8
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while (abs(energy_diff) > eps):
    energy = train_pqcnet().asnumpy()
    if iter_idx % 5 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy)))
    energy_diff = energy_last - energy
    energy_last = energy
    iter_idx += 1 """

print("Optimized energy: %20.16f" % (energy))