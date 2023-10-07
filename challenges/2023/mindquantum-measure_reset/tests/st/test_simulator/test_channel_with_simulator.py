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

"""Test channel with simulator."""
import numpy as np
from scipy.stats import entropy
import pytest

import mindquantum as mq
from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR

flip_and_damping_channel = [
    G.BitFlipChannel,
    G.PhaseFlipChannel,
    G.BitPhaseFlipChannel,
    G.AmplitudeDampingChannel,
    G.PhaseDampingChannel,
]

shots = 100000


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', list(SUPPORTED_SIMULATOR))
@pytest.mark.parametrize("channel", flip_and_damping_channel)
def test_flip_and_damping_channel(config, channel):
    """
    Description: Test flip and damping channel
    Expectation: success.
    """
    virtual_qc, dtype = config
    init_state = np.random.rand(2) + np.random.rand(2) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    p = np.random.rand()
    c = channel(p).on(0)
    tmp = np.outer(init_state, init_state.T.conj())
    ref_qs = np.zeros((2, 2), dtype=mq.to_np_type(dtype))
    for m in c.matrix():
        ref_qs += m @ tmp @ m.T.conj()
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    sim.set_qs(init_state)
    if virtual_qc.startswith("mqmatrix"):
        sim.apply_gate(c)
        assert np.allclose(sim.get_qs(), ref_qs)
    else:
        res = sim.sampling(Circuit([c, G.Measure().on(0)]), shots=shots)
        difference = entropy(np.array(list(res.data.values())) / shots, ref_qs.diagonal().real)
        assert difference < 1e-4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', list(SUPPORTED_SIMULATOR))
def test_pauli_channel(config):
    """
    Description: Test pauli channel
    Expectation: success.
    """
    virtual_qc, dtype = config
    init_state = np.random.rand(2) + np.random.rand(2) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    px, py, pz = np.random.rand(3) / 3
    c = G.PauliChannel(px, py, pz).on(0)
    tmp = np.outer(init_state, init_state.T.conj())
    ref_qs = np.zeros((2, 2), dtype=mq.to_np_type(dtype))
    for m in c.matrix():
        ref_qs += m @ tmp @ m.T.conj()
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    sim.set_qs(init_state)
    if virtual_qc.startswith("mqmatrix"):
        sim.apply_gate(c)
        assert np.allclose(sim.get_qs(), ref_qs)
    else:
        res = sim.sampling(Circuit([c, G.Measure().on(0)]), shots=shots)
        difference = entropy(np.array(list(res.data.values())) / shots, ref_qs.diagonal().real)
        assert difference < 1e-4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', list(SUPPORTED_SIMULATOR))
def test_depolarizing_channel(config):
    """
    Description: Test depolarizing channel
    Expectation: success.
    """
    virtual_qc, dtype = config
    for n in (1, 2, 3):
        p = np.random.rand() * 4**n / (4**n - 1)
        c = G.DepolarizingChannel(p, n).on(list(range(n)))
        dim = 2**n
        init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
        init_state = init_state / np.linalg.norm(init_state)
        tmp = np.outer(init_state, init_state.T.conj())
        ref_qs = np.zeros((dim, dim), dtype=mq.to_np_type(dtype))
        for m in c.matrix():
            ref_qs += m @ tmp @ m.T.conj()
        sim = Simulator(virtual_qc, n, dtype=dtype)
        sim.set_qs(init_state)
        if virtual_qc.startswith("mqmatrix"):
            sim.apply_gate(c)
            assert np.allclose(sim.get_qs(), ref_qs)
        else:
            res = sim.sampling(Circuit(c).measure_all(), shots=shots)
            difference = entropy(np.array(list(res.data.values())) / shots, ref_qs.diagonal().real)
            assert difference < 1e-4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', list(SUPPORTED_SIMULATOR))
def test_kraus_channel(config):
    """
    Description: Test kraus channel
    Expectation: success.
    """
    virtual_qc, dtype = config
    init_state = np.random.rand(2) + np.random.rand(2) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    p = np.random.rand() * 4 / 3
    dep = G.DepolarizingChannel(p).on(0)
    c = G.KrausChannel('depolarizing', dep.matrix()).on(0)
    tmp = np.outer(init_state, init_state.T.conj())
    ref_qs = np.zeros((2, 2), dtype=mq.to_np_type(dtype))
    for m in c.matrix():
        ref_qs += m @ tmp @ m.T.conj()
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    sim.set_qs(init_state)
    if virtual_qc.startswith("mqmatrix"):
        sim.apply_gate(c)
        assert np.allclose(sim.get_qs(), ref_qs)
    else:
        res = sim.sampling(Circuit([c, G.Measure().on(0)]), shots=shots)
        difference = entropy(np.array(list(res.data.values())) / shots, ref_qs.diagonal().real)
        assert difference < 1e-4
