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
        sim.set_qs(np.array([1 + 0.5j, 1 + 0.5j]))
        sim.apply_gate(C.PauliChannel(0.1, 0, 0).on(0))
        sim.apply_gate(C.PauliChannel(0, 0, 0.1).on(0))
        sim.apply_gate(C.PauliChannel(0, 0.1, 0).on(0))
        assert np.allclose(sim.get_qs(), np.array([[0.3 - 0.4j, 0.192 - 0.256j], [0.192 + 0.256j, 0.3 - 0.4j]]))
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
        sim1.set_qs(np.array([1 + 0.5j, 1 + 0.5j]))
        sim1.apply_gate(C.BitFlipChannel(0.1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([[0.3 - 0.4j, 0.3 - 0.32j], [0.3 + 0.32j, 0.3 - 0.4j]]))
        sim1.apply_gate(C.PhaseFlipChannel(0.1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([[0.3 - 0.4j, 0.24 - 0.256j], [0.24 + 0.256j, 0.3 - 0.4j]]))
        sim1.apply_gate(C.BitPhaseFlipChannel(0.1).on(0))
        assert np.allclose(sim1.get_qs(), np.array([[0.3 - 0.4j, 0.192 - 0.256j], [0.192 + 0.256j, 0.3 - 0.4j]]))
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
        sim2 = Simulator(backend, 1, dtype=dtype)
        sim2.set_qs(np.array([1 + 0.5j, 1 + 0.5j]))
        sim2.apply_gate(C.DepolarizingChannel(0.1).on(0))
        assert np.allclose(
            sim2.get_qs(), np.array([[0.3 - 0.4j, 0.26 - 0.34666667j], [0.26 + 0.34666667j, 0.3 - 0.4j]])
        )
    else:
        sim2 = Simulator(backend, 1)
        sim2.apply_gate(C.DepolarizingChannel(0).on(0))
        assert np.allclose(sim2.get_qs(), np.array([1.0 + 0.0j, 0.0 + 0.0j]))


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
        sim.set_qs(np.array([1 + 0.5j, 1 + 0.5j, 1 + 0.5j, 1 + 0.5j]))
        sim.apply_gate(C.AmplitudeDampingChannel(0.1).on(0))
        assert np.allclose(
            sim.get_qs(),
            np.array(
                [
                    [0.165 - 0.22j, 0.14230249 - 0.18973666j, 0.165 - 0.22j, 0.14230249 - 0.18973666j],
                    [0.14230249 + 0.18973666j, 0.135 - 0.18j, 0.14230249 - 0.18973666j, 0.135 - 0.18j],
                    [0.165 + 0.22j, 0.14230249 + 0.18973666j, 0.165 - 0.22j, 0.14230249 - 0.18973666j],
                    [0.14230249 + 0.18973666j, 0.135 + 0.18j, 0.14230249 + 0.18973666j, 0.135 - 0.18j],
                ]
            ),
        )
        sim2 = Simulator(backend, 2, dtype=dtype)
        sim2.set_qs(np.array([1 + 0.5j, 1 + 0.5j, 1 + 0.5j, 1 + 0.5j]))
        sim2.apply_gate(C.PhaseDampingChannel(0.1).on(0))
        assert np.allclose(
            sim2.get_qs(),
            np.array(
                [
                    [0.15 - 0.2j, 0.14230249 - 0.18973666j, 0.15 - 0.2j, 0.14230249 - 0.18973666j],
                    [0.14230249 + 0.18973666j, 0.15 - 0.2j, 0.14230249 - 0.18973666j, 0.15 - 0.2j],
                    [0.15 + 0.2j, 0.14230249 + 0.18973666j, 0.15 - 0.2j, 0.14230249 - 0.18973666j],
                    [0.14230249 + 0.18973666j, 0.15 + 0.2j, 0.14230249 + 0.18973666j, 0.15 - 0.2j],
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
        sim.set_qs(np.array([1 + 0.5j, 1 + 0.5j, 1 + 0.5j, 1 + 0.5j]))
        sim.apply_gate(kraus.on(0))
        assert np.allclose(
            sim.get_qs(),
            np.array(
                [
                    [0.1515 - 0.202j, 0.14924812 - 0.19899749j, 0.1515 - 0.202j, 0.14924812 - 0.19899749j],
                    [0.14924812 + 0.19899749j, 0.1485 - 0.198j, 0.14924812 - 0.19899749j, 0.1485 - 0.198j],
                    [0.1515 + 0.202j, 0.14924812 + 0.19899749j, 0.1515 - 0.202j, 0.14924812 - 0.19899749j],
                    [0.14924812 + 0.19899749j, 0.1485 + 0.198j, 0.14924812 + 0.19899749j, 0.1485 - 0.198j],
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
