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
# pylint: disable=too-many-locals,invalid-name
from math import exp

import numpy as np
import pytest
from scipy.stats import entropy

import mindquantum as mq
from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR
from mindquantum.utils import random_circuit

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
@pytest.mark.parametrize('config', list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
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
@pytest.mark.parametrize('config', list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
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
@pytest.mark.parametrize('config', list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
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
            assert difference < 1e-3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_kraus_channel(config):
    """
    Description: Test kraus channel
    Expectation: success.
    """
    virtual_qc, dtype = config
    init_state = np.random.rand(2) + np.random.rand(2) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    tmp = np.outer(init_state, init_state.T.conj())

    c0 = G.KrausChannel('ad', G.AmplitudeDampingChannel(np.random.rand()).matrix()).on(0)
    c1 = G.KrausChannel('dep', G.DepolarizingChannel(np.random.rand() * 4 / 3).matrix()).on(0)
    c2 = G.KrausChannel('gp', [1j * i for i in G.DepolarizingChannel(np.random.rand() * 4 / 3).matrix()]).on(0)

    for c in (c0, c1, c2):
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
@pytest.mark.parametrize('config', list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_grouped_pauli_channel(config):
    """
    Description: Test thermal relaxation channel
    Expectation: success.
    """
    virtual_qc, dtype = config
    n_qubits = 3
    old = random_circuit(n_qubits, 20)
    probs = np.random.uniform(0, 1, size=(n_qubits, 3))
    probs = probs * (0.8 / probs.sum(axis=1))[:, None]
    paulis = Circuit([G.PauliChannel(*i).on(idx) for idx, i in enumerate(probs)])
    grouped = Circuit([G.GroupedPauliChannel(probs).on(list(range(n_qubits)))])
    sim = Simulator(virtual_qc, n_qubits, dtype=dtype)
    circ1 = (old + paulis).measure_all()
    circ2 = (old + grouped).measure_all()
    if virtual_qc.startswith("mqmatrix"):
        sim.apply_circuit(circ1.remove_measure())
        qs1 = sim.get_qs()
        sim.reset()
        sim.apply_circuit(circ2.remove_measure())
        qs2 = sim.get_qs()
        assert np.allclose(qs1, qs2, atol=1e-4)
    else:
        res1 = sim.sampling(circ1, shots=shots)
        res2 = sim.sampling(circ2, shots=shots)
        difference = entropy(np.array(list(res1.data.values())) / shots, np.array(list(res2.data.values())) / shots)
        assert difference < 1e-3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_thermal_relaxation_channel(config):
    """
    Description: Test thermal relaxation channel
    Expectation: success.
    """
    virtual_qc, dtype = config
    init_state = np.random.rand(2) + np.random.rand(2) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    tmp = np.outer(init_state, init_state.T.conj())
    t1, t2, gate_time = np.random.rand(3) * 10000
    if t2 >= 2 * t1:
        with pytest.raises(ValueError):
            c = G.ThermalRelaxationChannel(t1, t2, gate_time).on(0)
        return
    e1 = exp(-gate_time / t1)
    e2 = exp(-gate_time / t2)
    c = G.ThermalRelaxationChannel(t1, t2, gate_time).on(0)
    choi_mat = np.array([[1, 0, 0, e2], [0, 0, 0, 0], [0, 0, 1 - e1, 0], [e2, 0, 0, e1]])
    mat = choi_mat @ np.kron(tmp.T, np.eye(2, 2))
    ref_dm = np.array([[mat[0][0] + mat[2][2], mat[0][1] + mat[2][1]], [mat[1][0] + mat[3][2], mat[1][1] + mat[3][3]]])
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    sim.set_qs(init_state)
    if virtual_qc.startswith("mqmatrix"):
        sim.apply_gate(c)
        assert np.allclose(sim.get_qs(), ref_dm)
    else:
        res = sim.sampling(Circuit([c, G.Measure().on(0)]), shots=shots)
        difference = entropy(np.array(list(res.data.values())) / shots, ref_dm.diagonal().real)
        assert difference < 1e-4
