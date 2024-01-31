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

import platform

import numpy as np
import pytest

_HAS_MINDSPORE = True
AVAILABLE_BACKEND = []
try:
    import mindspore as ms

    from mindquantum.core import gates as G
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.operators import Hamiltonian, QubitOperator
    from mindquantum.framework import MQLayer, QRamVecLayer
    from mindquantum.simulator import Simulator
    from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR

    AVAILABLE_BACKEND = list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR))

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
except ImportError:
    _HAS_MINDSPORE = False

    def get_supported_simulator():
        """Dummy function."""
        return []


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
def test_mindquantumlayer(config):
    """
    Description: Test MQLayer
    Expectation:
    """
    backend, dtype = config
    encoder = Circuit()
    ansatz = Circuit()
    encoder += G.RX('e1').on(0)
    encoder += G.RY('e2').on(1)
    ansatz += G.X.on(1, 0)
    ansatz += G.RY('p1').on(0)
    ham = Hamiltonian(QubitOperator('Z0').astype(dtype))
    ms.set_seed(55)
    circ = encoder.as_encoder() + ansatz.as_ansatz()
    sim = Simulator(backend, circ.n_qubits, dtype=dtype)
    f_g_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQLayer(f_g_ops)
    encoder_data = ms.Tensor(np.array([[0.1, 0.2]]).astype(np.float32))
    res = net(encoder_data)
    assert np.allclose(res.asnumpy()[0, 0], 0.994962, atol=1e-2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
@pytest.mark.skipif(platform.system() == "Darwin", reason='MacOS not work currently.')
def test_qram_vec_layer(config):
    """
    Description: Test QRamVec
    Expectation:
    """
    backend, dtype = config
    if backend == 'mqmatrix':
        return
    ms.set_seed(42)
    ans = Circuit().ry('a', 0).rx('b', 0).as_ansatz()
    ham = Hamiltonian(QubitOperator('Z0')).astype(dtype)
    sim = Simulator(backend, 1, dtype=dtype)
    quantum_state = np.array([[1.0, 2.0]]) / np.sqrt(5)
    qs_r, qs_i = ms.Tensor(quantum_state.real), ms.Tensor(quantum_state.imag)
    net = QRamVecLayer(ham, ans, sim, len(quantum_state))
    opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
    train_net = ms.nn.TrainOneStepCell(net, opti)
    for _ in range(100):
        train_net(qs_r, qs_i)
    weight = net.weight.asnumpy()
    assert np.allclose(weight, np.array([9.2439342e-01, -3.3963533e-04]), atol=1e-2)
