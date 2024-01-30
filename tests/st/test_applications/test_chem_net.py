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
from pathlib import Path

import pytest

_HAS_MINDSPORE = True
AVAILABLE_BACKEND = []

try:
    import mindspore as ms

    from mindquantum.algorithm.nisq import generate_uccsd
    from mindquantum.core import gates as G
    from mindquantum.core.circuit import Circuit
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


_HAS_OPENFERMION = True
try:
    # pylint: disable=unused-import
    from openfermion import FermionOperator as OFFermionOperator
except (ImportError, AttributeError):
    _HAS_OPENFERMION = False
_FORCE_TEST = bool(os.environ.get("FORCE_TEST", False))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
@pytest.mark.skipif(not _HAS_OPENFERMION, reason='openfermion is not installed')
@pytest.mark.skipif(not _FORCE_TEST, reason='Set not force test')
def test_vqe_net(config):  # pylint: disable=too-many-locals
    """
    Description: Test vqe
    Expectation:
    """
    backend, dtype = config
    if backend == 'mqmatrix':
        return
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
    (
        ansatz_circuit,
        _,  # init_amplitudes
        _,  # ansatz_parameter_names
        hamiltonian_qubitop,
        _,  # n_qubits
        n_electrons,
    ) = generate_uccsd(str(Path(__file__).parent.parent / 'H4.hdf5'), threshold=-1)
    hf_circuit = Circuit([G.X.on(i) for i in range(n_electrons)])
    vqe_circuit = hf_circuit + ansatz_circuit
    sim = Simulator(backend, vqe_circuit.n_qubits, dtype=dtype)
    f_g_ops = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_qubitop.real.astype(dtype)), vqe_circuit)
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
