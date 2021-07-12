import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import mindquantum as mq
from mindquantum import Circuit, X, RX, Hamiltonian
from mindquantum.circuit import generate_uccsd
from mindquantum.nn import generate_pqc_operator
import mindspore as ms
import mindspore.context as context
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

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

print("Hartree-Fock energy: %20.16f Ha" % (molecule_of.hf_energy))
print("CCSD energy: %20.16f Ha" % (molecule_of.ccsd_energy))
print("FCI energy: %20.16f Ha" % (molecule_of.fci_energy))

molecule_of.save()
molecule_file = molecule_of.filename
print(molecule_file)

hartreefock_wfn_circuit = Circuit(
    [X.on(i) for i in range(molecule_of.n_electrons)])
print(hartreefock_wfn_circuit)

ansatz_circuit, init_amplitudes, ansatz_parameter_names, hamiltonian_QubitOp, n_qubits, n_electrons = generate_uccsd(
    molecule_file, th=-1)

total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()
print("Number of parameters: %d" % (len(ansatz_parameter_names)))

molecule_pqc = generate_pqc_operator(["null"], ansatz_parameter_names,
                                     RX("null").on(0) + total_circuit,
                                     Hamiltonian(hamiltonian_QubitOp))

class PQCNet(ms.nn.Cell):
    def __init__(self, pqc):
        super(PQCNet, self).__init__()
        self.pqc = pqc
        self.weight = Parameter(initializer("Zeros",
                                            len(self.pqc.ansatz_params_names)),
                                name="weight")
        self.encoder_data_dummy = ms.Tensor([[0]], self.weight.dtype)

    def construct(self):
        energy, _, grads = self.pqc(self.encoder_data_dummy, self.weight)
        return energy


molecule_pqcnet = PQCNet(molecule_pqc)

initial_energy = molecule_pqcnet()
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))

optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(),
                          learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

eps = 1.e-8
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while (abs(energy_diff) > eps):
    energy_i = train_pqcnet().asnumpy()
    if iter_idx % 5 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1

print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Optimized amplitudes: \n", molecule_pqcnet.weight.asnumpy())

from mindquantum.hiqfermion.transforms import Transform
from mindquantum.hiqfermion.ucc import get_qubit_hamiltonian
from mindquantum.hiqfermion.ucc import uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes
from mindquantum.circuit import TimeEvolution
from mindquantum.nn.mindquantum_ansatz_only_layer import MindQuantumAnsatzOnlyLayer

hamiltonian_QubitOp = get_qubit_hamiltonian(molecule_of)

ucc_fermion_ops = uccsd_singlet_generator(molecule_of.n_qubits,
                                          molecule_of.n_electrons,
                                          anti_hermitian=True)

ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()

ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
ansatz_parameter_names = ansatz_circuit.para_name

total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()

init_amplitudes_ccsd = uccsd_singlet_get_packed_amplitudes(
    molecule_of.ccsd_single_amps, molecule_of.ccsd_double_amps,
    molecule_of.n_qubits, molecule_of.n_electrons)
init_amplitudes_ccsd = [
    init_amplitudes_ccsd[param_i] for param_i in ansatz_parameter_names
]

molecule_pqcnet = MindQuantumAnsatzOnlyLayer(
    ansatz_parameter_names, total_circuit,
    Hamiltonian(hamiltonian_QubitOp.real))

molecule_pqcnet.weight = Parameter(
    ms.Tensor(init_amplitudes_ccsd, molecule_pqcnet.weight.dtype))
initial_energy = molecule_pqcnet()
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))

optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(),
                          learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

print("eps: ", eps)
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while (abs(energy_diff) > eps):
    energy_i = train_pqcnet().asnumpy()
    if iter_idx % 5 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1

print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Optimized amplitudes: \n", molecule_pqcnet.weight.asnumpy())
