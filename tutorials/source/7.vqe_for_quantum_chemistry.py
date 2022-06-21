# -*- coding: utf-8 -*-
#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Example of VQE implementation for quantum chemistry."""

import mindspore as ms
import mindspore.context as context
from mindspore.common.parameter import Parameter
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from mindquantum import Circuit, Hamiltonian, Simulator, X
from mindquantum.algorithm import generate_uccsd
from mindquantum.algorithm.nisq.chem import (
    Transform,
    get_qubit_hamiltonian,
    uccsd_singlet_generator,
    uccsd_singlet_get_packed_amplitudes,
)
from mindquantum.core.operators import TimeEvolution
from mindquantum.framework import MQAnsatzOnlyLayer

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

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

hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(molecule_of.n_electrons)])
print(hartreefock_wfn_circuit)

ansatz_circuit, init_amplitudes, ansatz_parameter_names, hamiltonian_qubit_op, n_qubits, n_electrons = generate_uccsd(
    molecule_file, th=-1
)

total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()
print("Number of parameters: %d" % (len(ansatz_parameter_names)))

sim = Simulator('projectq', total_circuit.n_qubits)
molecule_pqc = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_qubit_op), total_circuit)


molecule_pqcnet = MQAnsatzOnlyLayer(molecule_pqc, 'Zeros')

initial_energy = molecule_pqcnet()
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))

optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

eps = 1.0e-8
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while abs(energy_diff) > eps:
    energy_i = train_pqcnet().asnumpy()
    if iter_idx % 5 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1

print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Optimized amplitudes: \n", molecule_pqcnet.weight.asnumpy())


hamiltonian_qubit_op = get_qubit_hamiltonian(molecule_of)

ucc_fermion_ops = uccsd_singlet_generator(molecule_of.n_qubits, molecule_of.n_electrons, anti_hermitian=True)

ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()

ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
ansatz_parameter_names = ansatz_circuit.params_name

total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()

init_amplitudes_ccsd = uccsd_singlet_get_packed_amplitudes(
    molecule_of.ccsd_single_amps, molecule_of.ccsd_double_amps, molecule_of.n_qubits, molecule_of.n_electrons
)
init_amplitudes_ccsd = [init_amplitudes_ccsd[param_i] for param_i in ansatz_parameter_names]

grad_ops = Simulator('projectq', total_circuit.n_qubits).get_expectation_with_grad(
    Hamiltonian(hamiltonian_qubit_op.real), total_circuit
)

molecule_pqcnet = MQAnsatzOnlyLayer(grad_ops)

molecule_pqcnet.weight = Parameter(ms.Tensor(init_amplitudes_ccsd, molecule_pqcnet.weight.dtype))
initial_energy = molecule_pqcnet()
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))

optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

print("eps: ", eps)
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while abs(energy_diff) > eps:
    energy_i = train_pqcnet().asnumpy()
    if iter_idx % 5 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1

print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Optimized amplitudes: \n", molecule_pqcnet.weight.asnumpy())
