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
"""Test unitary coupled-cluster ansatz"""

import os

import numpy as np
import pytest

os.environ.setdefault('OMP_NUM_THREADS', '8')

_HAS_MINDSPORE = True
try:
    import mindspore as ms

    from mindquantum.algorithm.nisq.chem import UCCAnsatz
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import X
    from mindquantum.core.operators import Hamiltonian, QubitOperator
    from mindquantum.framework import MQAnsatzOnlyLayer
    from mindquantum.simulator import Simulator

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
except ImportError:
    _HAS_MINDSPORE = False


@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
def test_uccsd():  # pylint: disable=too-many-locals
    """
    Description:
    Expectation:
    """
    # Hydrogen molecule
    ham = (
        QubitOperator("", (-0.5339363487727398 + 0j))
        + QubitOperator("X0 X1 Y2 Y3", (-0.0647846187202642 + 0j))
        + QubitOperator("X0 Y1 Y2 X3", (0.0647846187202642 + 0j))
        + QubitOperator("Y0 X1 X2 Y3", (0.0647846187202642 + 0j))
        + QubitOperator("Y0 Y1 X2 X3", (-0.0647846187202642 + 0j))
        + QubitOperator("Z0", (0.06727930458983417 + 0j))
        + QubitOperator("Z0 Z1", (0.12736570310657463 + 0j))
        + QubitOperator("Z0 Z2", (0.06501569581211997 + 0j))
        + QubitOperator("Z0 Z3", (0.12980031453238416 + 0j))
        + QubitOperator("Z1", (0.06727930458983417 + 0j))
        + QubitOperator("Z1 Z2", (0.12980031453238416 + 0j))
        + QubitOperator("Z1 Z3", (0.06501569581211997 + 0j))
        + QubitOperator("Z2", (0.006651295687574388 + 0j))
        + QubitOperator("Z2 Z3", (0.13366602988233994 + 0j))
        + QubitOperator("Z3", (0.006651295687574388 + 0j))
    )
    n_qubits = 4
    n_electrons = 2
    occ_orb = [0]
    vir_orb = [1]
    generalized = True
    trotter_step = 2
    ucc = UCCAnsatz(n_qubits, n_electrons, occ_orb, vir_orb, generalized, trotter_step)
    total_circuit = Circuit()
    for i in range(n_electrons):
        total_circuit += X.on(i)
    total_circuit += ucc.circuit
    sim = Simulator('projectq', total_circuit.n_qubits)
    f_g_ops = sim.get_expectation_with_grad(Hamiltonian(ham.real), total_circuit)
    net = MQAnsatzOnlyLayer(f_g_ops)
    opti = ms.nn.Adagrad(net.trainable_params(), learning_rate=4e-2)
    train_net = ms.nn.TrainOneStepCell(net, opti)
    for i in range(50):
        res = train_net().asnumpy()[0]
    assert np.allclose(round(res, 4), -0.9486)
