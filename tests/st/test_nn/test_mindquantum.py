# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test mindquantum."""

import os
os.environ['OMP_NUM_THREADS'] = '8'
from mindquantum.ops import QubitOperator
import numpy as np
import mindspore as ms
import mindquantum.gate as G
from mindquantum.nn import MindQuantumLayer, MindQuantumAnsatzOnlyLayer
from mindquantum.circuit import generate_uccsd
from mindquantum import Circuit, Hamiltonian


def test_mindquantumlayer_forward():
    """Test mindquantum layer."""
    encoder = Circuit()
    ansatz = Circuit()
    encoder += G.RX('e1').on(0)
    ansatz += G.RY('a').on(0)
    ham = Hamiltonian(QubitOperator('Z0'))
    ms.set_seed(42)
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")
    net = MindQuantumLayer(['e1'], ['a'], encoder + ansatz, ham)
    encoder_data = ms.Tensor(np.array([[0.5]]).astype(np.float32))
    res = net(encoder_data)
    assert round(float(res.asnumpy()[0, 0]), 3) == 0.878


def test_vqe_convergence():
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")
    ansatz_circuit, \
        init_amplitudes, \
        ansatz_parameter_names, \
        hamiltonian_qubitop, \
        n_qubits, n_electrons = generate_uccsd(
            './tests/st/H4.hdf5', th=-1)
    hf_circuit = Circuit([G.X.on(i) for i in range(n_electrons)])
    vqe_circuit = hf_circuit + ansatz_circuit
    molecule_pqcnet = MindQuantumAnsatzOnlyLayer(
        ansatz_parameter_names, vqe_circuit,
        Hamiltonian(hamiltonian_qubitop.real))
    optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(),
                              learning_rate=4e-2)
    train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)
    eps = 1e-8
    energy_diff = 1.
    energy_last = 1.
    iter_idx = 0
    iter_max = 100
    while (abs(energy_diff) > eps) and (iter_idx < iter_max):
        energy_i = train_pqcnet().asnumpy()
        energy_diff = energy_last - energy_i
        energy_last = energy_i
        iter_idx += 1

    assert round(energy_i.item(), 3) == -2.166
