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
"""Test max_2_sat"""

import numpy as np
import pytest

_HAS_MINDSPORE = True
AVAILABLE_BACKEND = []
try:
    import mindspore as ms

    from mindquantum.algorithm.nisq import Max2SATAnsatz
    from mindquantum.core.operators import Hamiltonian
    from mindquantum.framework import MQAnsatzOnlyLayer
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
def test_max_2_sat(config):
    """
    Description:
    Expectation:
    """
    backend, dtype = config
    clauses = [(1, 2), (1, -2), (-1, 2), (-1, -2), (1, 3)]
    depth = 3
    max2sat = Max2SATAnsatz(clauses, depth)
    sim = Simulator(backend, max2sat.circuit.n_qubits, dtype=dtype)
    ham = max2sat.hamiltonian.astype(dtype)
    f_g_ops = sim.get_expectation_with_grad(Hamiltonian(ham), max2sat.circuit)
    ms.set_seed(42)
    net = MQAnsatzOnlyLayer(f_g_ops)
    opt = ms.nn.Adagrad(net.trainable_params(), learning_rate=4e-1)
    train_net = ms.nn.TrainOneStepCell(net, opt)
    ret = 0
    for _ in range(100):
        ret = train_net().asnumpy()[0]
    assert np.allclose(round(ret, 3), 1)
