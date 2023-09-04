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
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Test method of mqvector simulator."""
import subprocess

import numpy as np
import pytest
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from scipy.linalg import logm, sqrtm

import mindquantum as mq
from mindquantum.core import gates as G
from mindquantum.core.circuit import UN, Circuit
from mindquantum.simulator import Simulator, fidelity
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR
from mindquantum.utils import random_circuit

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

try:
    importlib_metadata.import_module("numba")
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

_HAS_GPU = False

try:
    subprocess.check_output('nvidia-smi')
    _HAS_GPU = True
except FileNotFoundError:
    _HAS_GPU = False


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "virtual_qc",
    [
        'mqvector',
        pytest.param('mqvector_gpu', marks=pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')),
    ],
)
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_get_expectation(virtual_qc, dtype):
    """
    Description: test get expectation
    Expectation: success.
    """
    init_state = np.random.rand(8) + np.random.rand(8) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    init_state_left = np.random.rand(8) + np.random.rand(8) * 1j
    init_state_left = init_state_left / np.linalg.norm(init_state_left)
    circ = random_circuit(3, 100)
    circ_left = random_circuit(3, 100)
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    sim_left = Simulator(virtual_qc, 3, dtype=dtype)
    sim.set_qs(init_state)
    sim_left.set_qs(init_state_left)
    ham0 = Hamiltonian(QubitOperator('X0 Y1'), dtype=dtype)
    ham1 = ham0.sparse(3)
    ham2 = Hamiltonian(csr_matrix(ham0.hamiltonian.matrix(3)), dtype=dtype)
    for ham in (ham0, ham1, ham2):
        f = sim.get_expectation(ham, circ)
        ref_f = (
            init_state.T.conj() @ circ.hermitian().matrix() @ ham0.hamiltonian.matrix(3) @ circ.matrix() @ init_state
        )
        assert np.allclose(f, ref_f, atol=1e-6)

        f = sim.get_expectation(ham, circ, circ_left)
        ref_f = (
            init_state.T.conj()
            @ circ_left.hermitian().matrix()
            @ ham0.hamiltonian.matrix(3)
            @ circ.matrix()
            @ init_state
        )
        assert np.allclose(f, ref_f, atol=1e-6)

        f = sim.get_expectation(ham, circ, simulator_left=sim_left)
        ref_f = (
            init_state_left.T.conj()
            @ circ.hermitian().matrix()
            @ ham0.hamiltonian.matrix(3)
            @ circ.matrix()
            @ init_state
        )
        assert np.allclose(f, ref_f, atol=1e-6)

        f = sim.get_expectation(ham, circ, circ_left, sim_left)
        ref_f = (
            init_state_left.T.conj()
            @ circ_left.hermitian().matrix()
            @ ham0.hamiltonian.matrix(3)
            @ circ.matrix()
            @ init_state
        )
        assert np.allclose(f, ref_f, atol=1e-6)


single_parameter_gate = [
    G.RX,
    G.RY,
    G.RZ,
    # G.SWAPalpha,
    G.XX,
    G.YY,
    G.ZZ,
    G.GlobalPhase,
    G.PhaseShift,
    G.Rxx,
    G.Rxy,
    G.Rxz,
    G.Ryy,
    G.Ryz,
    G.Rzz,
]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "virtual_qc",
    [
        'mqvector',
        pytest.param('mqvector_gpu', marks=pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')),
    ],
)
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_parameter_shift_rule(virtual_qc, dtype):  # pylint: disable=too-many-locals
    """
    Description:
    Expectation:
    """
    circ = random_circuit(3, 10)
    for i, gate in enumerate(single_parameter_gate):
        gate_ = gate({f'pr0_{i}': 1, f'pr1_{i}': 2})
        circ += gate_.on(list(range(gate_.n_qubits)))
        circ += random_circuit(3, 10)
    circ += G.U3('u3_theta', 'u3_phi', 'u3_lamda').on(0)
    circ += random_circuit(3, 10)
    circ += G.FSim('fsim_theta', 'fsim_phi').on([0, 1])
    circ += random_circuit(3, 10)
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    ham = Hamiltonian(QubitOperator('X0') + QubitOperator('Z0'), dtype=dtype)
    grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=True)
    pr = np.random.rand(len(circ.all_paras))
    f, g = grad_ops(pr)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ)
    ref_f, ref_g = ref_grad_ops(pr)
    assert np.allclose(f, ref_f, atol=1e-6)
    assert np.allclose(g, ref_g, atol=1e-6)
