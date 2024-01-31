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

# pylint: disable=invalid-name
"""Test quantum fisher information."""
import numpy as np
import pytest

from mindquantum.core.circuit import (
    Circuit,
    partial_psi_partial_psi,
    partial_psi_psi,
    qfi,
)
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.simulator import Simulator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR

AVAILABLE_BACKEND = list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
def test_qfi(config):
    """
    Description: Test qfi
    Expectation: success
    """
    # pylint: disable=too-many-locals
    backend, dtype = config
    if backend == 'mqmatrix':
        return
    a = PR('a')
    b = PR('b')
    val = PR({'a': 1, 'b': 2})
    circ = Circuit().rx(a, 0).ry(b, 0)
    sim = Simulator(backend, 1, dtype=dtype)
    sim.apply_gate(circ[0], val, True)
    sim.apply_gate(circ[1], val, False)
    partial_psi_a = sim.get_qs()

    sim.reset()
    sim.apply_gate(circ[0], val, False)
    sim.apply_gate(circ[1], val, True)
    partial_psi_b = sim.get_qs()

    sim.reset()
    sim.apply_gate(circ[0], val, False)
    sim.apply_gate(circ[1], val, False)
    qs = sim.get_qs()

    m_pppp_exp = np.zeros((2, 2), np.complex128)
    m_pppp_exp[0, 0] = np.vdot(partial_psi_a, partial_psi_a)
    m_pppp_exp[0, 1] = np.vdot(partial_psi_a, partial_psi_b)
    m_pppp_exp[1, 0] = np.vdot(partial_psi_b, partial_psi_a)
    m_pppp_exp[1, 1] = np.vdot(partial_psi_b, partial_psi_b)
    m_pppp_m = partial_psi_partial_psi(circ)(val)
    assert np.allclose(m_pppp_exp, m_pppp_m)

    m_ppp_exp = np.zeros(2, np.complex128)
    m_ppp_exp[0] = np.vdot(partial_psi_a, qs)
    m_ppp_exp[1] = np.vdot(partial_psi_b, qs)
    m_ppp_m = partial_psi_psi(circ)(val)
    assert np.allclose(m_ppp_exp, m_ppp_m)

    qfi_exp = np.real(m_pppp_exp - np.outer(m_ppp_exp, np.conj(m_ppp_exp))) * 4
    qfi_m = qfi(circ)(val)
    assert np.allclose(qfi_exp, qfi_m, atol=1e-5)
