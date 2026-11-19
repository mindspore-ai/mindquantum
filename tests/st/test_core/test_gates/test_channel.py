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
"""Test channel."""

from math import exp, sqrt

import numpy as np
import pytest

import mindquantum.core.gates.channel as C  # noqa: N812

I = np.array([[1.0 + 0.0j, 0], [0, 1]])  # noqa: E741
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def test_pauli_channel():
    """
    Description: Test pauli channel
    Expectation: success.
    """
    px, py, pz = np.random.rand(3) / 3
    assert np.allclose(
        C.PauliChannel(px, py, pz).matrix(), [sqrt(1 - px - py - pz) * I, sqrt(px) * X, sqrt(py) * Y, sqrt(pz) * Z]
    )
    assert np.allclose(C.BitFlipChannel(px).matrix(), [sqrt(1 - px) * I, sqrt(px) * X])
    assert np.allclose(C.PhaseFlipChannel(pz).matrix(), [sqrt(1 - pz) * I, sqrt(pz) * Z])
    assert np.allclose(C.BitPhaseFlipChannel(py).matrix(), [sqrt(1 - py) * I, sqrt(py) * Y])


def test_depolarizing_channel():
    """
    Description: Test depolarizing channel
    Expectation: success.
    """
    for n in (1, 2, 3):
        p = np.random.rand() * 4**n / (4**n - 1)
        mat_list = [I, X, Y, Z]
        for _ in range(n - 1):
            tmp = []
            for j in (I, X, Y, Z):
                for k in mat_list:
                    tmp.append(np.kron(k, j))
            mat_list = tmp
        mat_list[0] = sqrt(1 - p * (4**n - 1) / 4**n) * mat_list[0]
        for i in range(1, len(mat_list)):
            mat_list[i] = sqrt(p) / (2**n) * mat_list[i]
        assert np.allclose(C.DepolarizingChannel(p, n).matrix(), mat_list)


def test_damping_channel():
    """
    Description: Test damping channel
    Expectation: success.
    """
    gamma = np.random.rand()
    mat_0 = np.array([[1, 0], [0, sqrt(1 - gamma)]])
    mat_1 = np.array([[0, sqrt(gamma)], [0, 0]])
    assert np.allclose(C.AmplitudeDampingChannel(gamma).matrix(), [mat_0, mat_1])
    mat_1 = np.array([[0, 0], [0, sqrt(gamma)]])
    assert np.allclose(C.PhaseDampingChannel(gamma).matrix(), [mat_0, mat_1])


def test_kraus_channel():
    """
    Description: Test kraus channel
    Expectation: success.
    """
    gamma = np.random.rand()
    mat_0 = np.array([[1, 0], [0, sqrt(1 - gamma)]])
    mat_1 = np.array([[0, sqrt(gamma)], [0, 0]])
    assert np.allclose(C.KrausChannel('AD', [mat_0, mat_1]).matrix(), [mat_0, mat_1])


@pytest.mark.parametrize('t2', [35, 100, 125])
def test_thermal_relaxation_channel(t2):
    """
    Feature: Thermal relaxation channel Kraus operators.
    Description: Reconstruct the Choi matrix across T2 < T1, T2 == T1, and T1 < T2 < 2 * T1.
    Expectation: Reconstructed Choi matrix matches T1 population decay and T2 coherence decay.
    """
    t1, gate_time = 100, 90
    e1 = exp(-gate_time / t1)
    e2 = exp(-gate_time / t2)
    ref_choi = np.array([[1, 0, 0, e2], [0, 0, 0, 0], [0, 0, 1 - e1, 0], [e2, 0, 0, e1]])
    kraus = C.ThermalRelaxationChannel(t1, t2, gate_time).matrix()
    choi = sum(np.outer(k.T.ravel(), k.T.conj().ravel()) for k in kraus)
    np.testing.assert_allclose(choi, ref_choi, rtol=0, atol=1e-12)


@pytest.mark.parametrize('t2', [200, 201])
def test_thermal_relaxation_rejects_invalid_t2(t2):
    """
    Feature: Thermal relaxation channel parameter validation.
    Description: Construct ThermalRelaxationChannel with T2 >= 2 * T1.
    Expectation: Raise ValueError.
    """
    with pytest.raises(ValueError, match=r'T2 >= 2 \* T1'):
        C.ThermalRelaxationChannel(100, t2, 90)


def test_channel_with_ctrl_qubits():
    """
    Description: Test raise channel have control qubits
    Expectation:
    """
    with pytest.raises(Exception, match=r"cannot have control qubits"):
        C.PauliChannel(0.1, 0.2, 0.3).on(0, 1)
    with pytest.raises(Exception, match=r"cannot have control qubits"):
        C.AmplitudeDampingChannel(0.1).on(0, 1)
    with pytest.raises(Exception, match=r"cannot have control qubits"):
        C.PhaseDampingChannel(0.1).on(0, 1)
    with pytest.raises(Exception, match=r"cannot have control qubits"):
        C.DepolarizingChannel(0.1).on(0, 1)
    with pytest.raises(Exception, match=r"cannot have control qubits"):
        C.KrausChannel('kraus', C.DepolarizingChannel(0.1).matrix()).on(0, 1)
    with pytest.raises(Exception, match=r"cannot have control qubits"):
        C.ThermalRelaxationChannel(10000, 9000, 30).on(0, 1)
    with pytest.raises(Exception, match=r"cannot have control qubits"):
        C.GroupedPauliChannel(np.array([[0.1, 0.2, 0.3]])).on(0, 1)
