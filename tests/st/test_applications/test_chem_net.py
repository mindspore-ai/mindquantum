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

"""Test VQE."""

import os

import pytest

os.environ.setdefault('OMP_NUM_THREADS', '8')

_HAS_MINDSPORE = True
try:
    import mindspore as ms

    from mindquantum import Circuit, Hamiltonian, Simulator
    from mindquantum.algorithm.nisq.chem import generate_uccsd
    from mindquantum.core import gates as G
    from mindquantum.framework import MQAnsatzOnlyLayer

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
except ImportError:
    _HAS_MINDSPORE = False


@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
def test_vqe_net():  # pylint: disable=too-many-locals
    """
    Description: Test vqe
    Expectation:
    """
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
    (
        ansatz_circuit,
        _,  # init_amplitudes
        _,  # ansatz_parameter_names
        hamiltonian_qubitop,
        _,  # n_qubits
        n_electrons,
    ) = generate_uccsd('./tests/st/H4.hdf5', threshold=-1)
    hf_circuit = Circuit([G.X.on(i) for i in range(n_electrons)])
    vqe_circuit = hf_circuit + ansatz_circuit
    sim = Simulator('projectq', vqe_circuit.n_qubits)
    f_g_ops = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_qubitop.real), vqe_circuit)
    molecule_pqcnet = MQAnsatzOnlyLayer(f_g_ops)
    optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
    train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)
    eps = 1e-8
    energy_diff = 1.0
    energy_last = 1.0
    iter_idx = 0
    iter_max = 100
    while (abs(energy_diff) > eps) and (iter_idx < iter_max):
        energy_i = train_pqcnet().asnumpy()
        energy_diff = energy_last - energy_i
        energy_last = energy_i
        iter_idx += 1

    assert round(energy_i.item(), 3) == -2.166
