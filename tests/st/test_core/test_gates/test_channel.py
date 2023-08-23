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
import numpy as np
import pytest

import mindquantum.core.gates.channel as C
from mindquantum.core.gates import X
from mindquantum.simulator import Simulator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR

AVAILABLE_BACKEND = list(SUPPORTED_SIMULATOR)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
def test_pauli_channel(config):
    """
    Description: Test pauli channel
    Expectation: success.
    """

    backend, dtype = config
    sim = Simulator(backend, 1, dtype=dtype)
    if backend == "mqmatrix":
        sim.set_qs(np.array([1 + 0.5j, 1 - 0.5j]))
        sim.apply_gate(C.PauliChannel(0.1, 0, 0).on(0))
        sim.apply_gate(C.PauliChannel(0, 0, 0.1).on(0))
        sim.apply_gate(C.PauliChannel(0, 0.1, 0).on(0))
        assert np.allclose(sim.get_qs(), np.array([[0.5 - 0.0j, 0.192 + 0.256j], [0.192 - 0.256j, 0.5 - 0.0j]]))
    else:
        sim.apply_gate(C.PauliChannel(1, 0, 0).on(0))
        sim.apply_gate(C.PauliChannel(0, 0, 1).on(0))
        sim.apply_gate(C.PauliChannel(0, 1, 0).on(0))
        assert np.allclose(sim.get_qs(), np.array([0.0 + 1.0j, 0.0 + 0.0j]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
def test_flip_channel(config):
    """
    Description: Test flip channel
    Expectation: success.
    """

    backend, dtype = config
    sim1 = Simulator(backend, 1, dtype=dtype)
    if backend == "mqmatrix":
        sim1.set_qs(np.array([1 + 0.5j, 1 - 0.5j]))
        sim1.apply_gate(C.BitFlipChannel(0.1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([[0.5 - 0.0j, 0.3 + 0.32j], [0.3 - 0.32j, 0.5 - 0.0j]]))
        sim1.apply_gate(C.PhaseFlipChannel(0.1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([[0.5 - 0.0j, 0.24 + 0.256j], [0.24 - 0.256j, 0.5 - 0.0j]]))
        sim1.apply_gate(C.BitPhaseFlipChannel(0.1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([[0.5 - 0.0j, 0.192 + 0.256j], [0.192 - 0.256j, 0.5 - 0.0j]]))
    else:
        sim1.apply_gate(C.BitFlipChannel(1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([0.0 + 0.0j, 1.0 + 0.0j]))
        sim1.apply_gate(C.PhaseFlipChannel(1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([0.0 + 0.0j, -1.0 + 0.0j]))
        sim1.apply_gate(C.BitPhaseFlipChannel(1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([0.0 + 1.0j, 0.0 + 0.0j]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
def test_depolarizing_channel(config):
    """
    Description: Test depolarizing channel
    Expectation: success.
    """
    backend, dtype = config
    if backend == "mqmatrix":
        sim = Simulator(backend, 1, dtype=dtype)
        sim.set_qs(np.array([1 + 0.5j, 1 - 0.5j]))
        sim.apply_gate(C.DepolarizingChannel(0.1).on(0))
        assert np.allclose(sim.get_qs(), np.array([[0.5 - 0.0j, 0.27 + 0.36j], [0.27 - 0.36j, 0.5 - 0.0j]]))
        sim2 = Simulator(backend, 2, dtype=dtype)
        sim2.set_qs(np.array([1 + 0.5j, 2 + 0.5j, 3 + 0.5j, 4 + 0.5j]))
        sim2.apply_gate(C.DepolarizingChannel(0.1, 2).on([0, 1]))
        assert np.allclose(
            sim2.get_qs(),
            np.array(
                [
                    [0.06129032 - 0.0j, 0.06532258 + 0.01451613j, 0.09435484 + 0.02903226j, 0.1233871 + 0.04354839j],
                    [0.06532258 - 0.01451613j, 0.1483871 - 0.0j, 0.18145161 + 0.01451613j, 0.23951613 + 0.02903226j],
                    [0.09435484 - 0.02903226j, 0.18145161 - 0.01451613j, 0.29354839 - 0.0j, 0.35564516 + 0.01451613j],
                    [0.1233871 - 0.04354839j, 0.23951613 - 0.02903226j, 0.35564516 - 0.01451613j, 0.49677419 - 0.0j],
                ]
            ),
        )
    else:
        sim = Simulator(backend, 1, seed=42)
        sim.set_qs(np.array([1 + 0.5j, 1 - 0.5j]))
        sim.apply_gate(C.DepolarizingChannel(0.5).on(0))
        assert np.allclose(sim.get_qs(), np.array([0.63245553 + 0.31622777j, 0.63245553 - 0.31622777j]))
        sim2 = Simulator(backend, 2, seed=42)
        sim2.set_qs(np.array([1 + 0.5j, 2 + 0.5j, 3 + 0.5j, 4 + 0.5j]))
        sim2.apply_gate(C.DepolarizingChannel(0.5, 2).on([0, 1]))
        assert np.allclose(
            sim2.get_qs(),
            np.array(
                [0.1796053 + 0.08980265j, 0.3592106 + 0.08980265j, -0.53881591 - 0.08980265j, -0.71842121 - 0.08980265j]
            ),
        )


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
def test_damping_channel(config):
    """
    Description: Test damping channel
    Expectation: success.
    """
    backend, dtype = config
    if backend == "mqmatrix":
        sim = Simulator(backend, 2, dtype=dtype)
        sim.set_qs(np.array([1 + 0.5j, 1 - 0.5j, 2 + 0.5j, 2 - 0.5j]))
        sim.apply_gate(C.AmplitudeDampingChannel(0.1).on(0))
        assert np.allclose(
            sim.get_qs(),
            np.array(
                [
                    [0.125 - 0.0j, 0.06468295 + 0.08624394j, 0.225 + 0.04090909j, 0.15092689 + 0.1293659j],
                    [0.06468295 - 0.08624394j, 0.10227273 - 0.0j, 0.15092689 - 0.1293659j, 0.18409091 - 0.04090909j],
                    [0.225 - 0.04090909j, 0.15092689 + 0.1293659j, 0.425 - 0.0j, 0.32341476 + 0.17248787j],
                    [0.15092689 - 0.1293659j, 0.18409091 + 0.04090909j, 0.32341476 - 0.17248787j, 0.34772727 - 0.0j],
                ]
            ),
        )
        sim2 = Simulator(backend, 2, dtype=dtype)
        sim2.set_qs(np.array([1 + 0.5j, 1 - 0.5j, 2 + 0.5j, 2 - 0.5j]))
        sim2.apply_gate(C.PhaseDampingChannel(0.1).on(0))
        assert np.allclose(
            sim2.get_qs(),
            np.array(
                [
                    [0.11363636 - 0.0j, 0.06468295 + 0.08624394j, 0.20454545 + 0.04545455j, 0.15092689 + 0.1293659j],
                    [0.06468295 - 0.08624394j, 0.11363636 - 0.0j, 0.15092689 - 0.1293659j, 0.20454545 - 0.04545455j],
                    [0.20454545 - 0.04545455j, 0.15092689 + 0.1293659j, 0.38636364 - 0.0j, 0.32341476 + 0.17248787j],
                    [0.15092689 - 0.1293659j, 0.20454545 + 0.04545455j, 0.32341476 - 0.17248787j, 0.38636364 - 0.0j],
                ]
            ),
        )
    else:
        sim = Simulator(backend, 2, dtype=dtype)
        sim.apply_gate(X.on(0))
        sim.apply_gate(X.on(1))
        sim.apply_gate(C.AmplitudeDampingChannel(1).on(0))
        assert np.allclose(sim.get_qs(), np.array([0, 0, 1, 0]))
        sim2 = Simulator(backend, 2, dtype=dtype)
        sim2.apply_gate(X.on(0))
        sim2.apply_gate(X.on(1))
        sim2.apply_gate(C.PhaseDampingChannel(0.5).on(0))
        assert np.allclose(sim2.get_qs(), np.array([0, 0, 0, 1]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
def test_kraus_channel(config):
    """
    Description: Test kraus channel
    Expectation: success.
    """
    backend, dtype = config
    if backend == "mqmatrix":
        kmat0 = [[1, 0], [0, np.sqrt(0.99)]]
        kmat1 = [[0, 0.1], [0, 0]]
        kraus = C.KrausChannel("amplitude_damping", [kmat0, kmat1])
        sim = Simulator(backend, 2, dtype=dtype)
        sim.set_qs(np.array([1 + 0.5j, 1 - 0.5j, 2 + 0.5j, 2 - 0.5j]))
        sim.apply_gate(kraus.on(0))
        assert np.allclose(
            sim.get_qs(),
            np.array(
                [
                    [0.11477273 - 0.0j, 0.06784005 + 0.0904534j, 0.20659091 + 0.045j, 0.15829346 + 0.13568011j],
                    [0.06784005 - 0.0904534j, 0.1125 - 0.0j, 0.15829346 - 0.13568011j, 0.2025 - 0.045j],
                    [0.20659091 - 0.045j, 0.15829346 + 0.13568011j, 0.39022727 - 0.0j, 0.33920026 + 0.18090681j],
                    [0.15829346 - 0.13568011j, 0.2025 + 0.045j, 0.33920026 - 0.18090681j, 0.3825 - 0.0j],
                ]
            ),
        )
    else:
        kmat0 = [[1, 0], [0, 0]]
        kmat1 = [[0, 1], [0, 0]]
        kraus = C.KrausChannel("amplitude_damping", [kmat0, kmat1])
        sim = Simulator(backend, 2, dtype=dtype)
        sim.apply_gate(X.on(0))
        sim.apply_gate(X.on(1))
        sim.apply_gate(kraus.on(0))
        assert np.allclose(sim.get_qs(), np.array([0, 0, 1, 0]))
