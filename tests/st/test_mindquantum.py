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

from mindquantum.ops import QubitOperator
import numpy as np
import mindspore as ms
import mindquantum.gate as G
from mindquantum.nn import MindQuantumLayer
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
