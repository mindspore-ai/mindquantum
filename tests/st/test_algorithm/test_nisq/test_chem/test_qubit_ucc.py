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

_HAS_MINDSPORE = True
AVAILABLE_BACKEND = []

try:
    import mindspore as ms

    from mindquantum.algorithm.nisq import QubitUCCAnsatz
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import X
    from mindquantum.core.operators import Hamiltonian, QubitOperator
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


os.environ.setdefault('OMP_NUM_THREADS', '8')


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
def test_quccsd(config):  # pylint: disable=too-many-locals
    """
    Description:
    Expectation:
    """
    backend, dtype = config
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
    generalized = False
    trotter_step = 4
    ucc = QubitUCCAnsatz(n_qubits, n_electrons, occ_orb, vir_orb, generalized, trotter_step)
    total_circuit = Circuit()
    for i in range(n_electrons):
        total_circuit += X.on(i)
    total_circuit += ucc.circuit

    sim = Simulator(backend, total_circuit.n_qubits, dtype=dtype)
    f_g_ops = sim.get_expectation_with_grad(Hamiltonian(ham.real.astype(dtype)), total_circuit)
    net = MQAnsatzOnlyLayer(f_g_ops)
    opti = ms.nn.Adagrad(net.trainable_params(), learning_rate=4e-2)
    train_net = ms.nn.TrainOneStepCell(net, opti)
    for i in range(100):
        res = train_net().asnumpy()[0]
    assert np.allclose(round(res, 4), -0.9486, atol=1e-3)
