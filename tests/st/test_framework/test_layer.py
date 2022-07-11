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
"""Test QNN layers."""

import numpy as np
import pytest

_HAS_MINDSPORE = True
try:
    import mindspore as ms

    from mindquantum.core import gates as G
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.operators import Hamiltonian, QubitOperator
    from mindquantum.framework import MQLayer
    from mindquantum.simulator import Simulator

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
except ImportError:
    _HAS_MINDSPORE = False


@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
def test_mindquantumlayer():
    """
    Description: Test MQLayer
    Expectation:
    """
    encoder = Circuit()
    ansatz = Circuit()
    encoder += G.RX('e1').on(0)
    encoder += G.RY('e2').on(1)
    ansatz += G.X.on(1, 0)
    ansatz += G.RY('p1').on(0)
    ham = Hamiltonian(QubitOperator('Z0'))
    ms.set_seed(55)
    circ = encoder.as_encoder() + ansatz.as_ansatz()
    sim = Simulator('projectq', circ.n_qubits)
    f_g_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQLayer(f_g_ops)
    encoder_data = ms.Tensor(np.array([[0.1, 0.2]]).astype(np.float32))
    res = net(encoder_data)
    assert round(float(res.asnumpy()[0, 0]), 6) == round(float(0.9949919), 6)
