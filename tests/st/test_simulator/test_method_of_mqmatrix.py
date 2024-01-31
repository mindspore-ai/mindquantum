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

# pylint: disable=invalid-name
"""Test method of mqmatrix simulator."""

import numpy as np
import pytest
from scipy.linalg import logm, sqrtm
from scipy.sparse import csr_matrix
from scipy.stats import entropy

import mindquantum as mq
from mindquantum.core import gates as G
from mindquantum.core.circuit import UN, Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator, fidelity
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR
from mindquantum.utils import random_circuit


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", ['mqmatrix'])
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_set_qs_and_dm(virtual_qc, dtype):
    """
    Description: test setting density matrix
    Expectation: success.
    """
    qs = np.random.rand(2) + np.random.rand(2) * 1j
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    sim.set_qs(qs)
    qs = qs / np.linalg.norm(qs)
    assert np.allclose(sim.get_qs(), np.outer(qs, qs.conj()))

    qs2 = np.random.rand(2) + np.random.rand(2) * 1j
    dm = np.outer(qs, qs.conj()) + np.outer(qs2, qs2.conj())
    sim.reset()
    sim.set_qs(dm)
    dm = dm / np.trace(dm)
    assert np.allclose(sim.get_qs(), dm, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", ['mqmatrix'])
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_get_partial_trace(virtual_qc, dtype):
    """
    Description: test partial trace of density matrix
    Expectation: success.
    """
    circ = random_circuit(3, 100)
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    sim.apply_circuit(circ)
    mat = sim.get_partial_trace([0, 1])
    qs = sim.get_qs()
    ref_mat = np.array(
        [[np.trace(qs[0:4, 0:4]), np.trace(qs[0:4, 4:8])], [np.trace(qs[4:8, 0:4]), np.trace(qs[4:8, 4:8])]]
    )
    assert np.allclose(mat, ref_mat, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", ['mqmatrix'])
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_purity(virtual_qc, dtype):
    """
    Description: test purity of density matrix
    Expectation: success.
    """
    circ = random_circuit(3, 100)
    circ = circ.with_noise(G.DepolarizingChannel(0.1))
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    sim.apply_circuit(circ)
    purity = sim.purity()
    ref_purity = np.trace(sim.get_qs() @ sim.get_qs())
    assert np.allclose(purity, ref_purity, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", ['mqmatrix'])
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_get_pure_state_vector(virtual_qc, dtype):
    """
    Description: test get pure state vector from density matrix
    Expectation: success.
    """
    init_state = np.random.rand(8) + np.random.rand(8) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    dm = np.outer(init_state, init_state.conj())
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    sim.set_qs(dm)
    state_vector = sim.get_pure_state_vector()
    assert np.allclose(np.outer(state_vector, state_vector.conj()), sim.get_qs(), atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_apply_hamiltonian(config):
    """
    Description: test apply hamiltonian
    Expectation: success.
    """
    virtual_qc, dtype = config
    circ = random_circuit(3, 100)
    ham0 = Hamiltonian(QubitOperator('X0 Y1'), dtype=dtype)
    ham1 = ham0.sparse(3)
    ham2 = Hamiltonian(csr_matrix(ham0.hamiltonian.matrix(3)), dtype=dtype)
    for ham in (ham0, ham1, ham2):
        sim = Simulator(virtual_qc, 3, dtype=dtype)
        sim.apply_circuit(circ)
        sim.apply_hamiltonian(ham)
        qs = sim.get_qs()
        sim.reset()
        sim.apply_circuit(circ)
        sim.apply_gate(G.X.on(0))
        sim.apply_gate(G.Y.on(1))
        ref_qs = sim.get_qs()
        assert np.allclose(qs, ref_qs, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_sampling(config):
    """
    Description: test sampling
    Expectation: success.
    """
    virtual_qc, dtype = config
    shots = 10000
    qs = np.random.rand(4) + np.random.rand(4) * 1j
    sim = Simulator(virtual_qc, 2, dtype=dtype)
    sim.set_qs(qs)
    res = sim.sampling(Circuit(UN(G.Measure(), [0, 1])), shots=shots)
    if virtual_qc.startswith("mqmatrix"):
        ref_distribution = sim.get_qs().diagonal().real
    else:
        ref_distribution = np.abs(qs) ** 2
    nonzero = []
    for key in res.data.keys():
        i = int(key, 2)
        nonzero.append(ref_distribution[i])
    difference = entropy(np.array(list(res.data.values())) / shots, nonzero)
    assert difference < 1e-3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_get_expectation(config):
    """
    Description: test get expectation
    Expectation: success.
    """
    virtual_qc, dtype = config
    init_state = np.random.rand(8) + np.random.rand(8) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    circ = random_circuit(3, 100)
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    sim.set_qs(init_state)
    ham0 = Hamiltonian(QubitOperator('X0 Y1') + QubitOperator('Z0'), dtype=dtype)
    ham1 = ham0.sparse(3)
    ham2 = Hamiltonian(csr_matrix(ham0.hamiltonian.matrix(3)), dtype=dtype)
    for ham in (ham0, ham1, ham2):
        f = sim.get_expectation(ham, circ)
        ref_f = (
            init_state.T.conj() @ circ.hermitian().matrix() @ ham0.hamiltonian.matrix(3) @ circ.matrix() @ init_state
        )
        assert np.allclose(f, ref_f, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_get_expectation_with_grad(config):
    """
    Description: test get expectation with gradient
    Expectation: success.
    """
    # pylint: disable=too-many-locals
    virtual_qc, dtype = config
    init_state = np.random.rand(8) + np.random.rand(8) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    circ0 = random_circuit(3, 100)
    circ1 = random_circuit(3, 100)
    pr_gate = G.RX({'a': 1, 'b': 2}).on(0)
    circ = circ0 + pr_gate + circ1
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    sim.set_qs(init_state)
    ham0 = Hamiltonian(QubitOperator('X0 Y1') + QubitOperator('Z0'), dtype=dtype)
    ham1 = ham0.sparse(3)
    ham2 = Hamiltonian(csr_matrix(ham0.hamiltonian.matrix(3)), dtype=dtype)
    for ham in (ham0, ham1, ham2):
        grad_ops = sim.get_expectation_with_grad(ham, circ)
        pr = np.random.rand(2) * 2 * np.pi
        f, g = grad_ops(pr)
        ref_f = (
            init_state.T.conj()
            @ circ.hermitian().matrix({'a': pr[0], 'b': pr[1]})
            @ ham0.hamiltonian.matrix(3)
            @ circ.matrix({'a': pr[0], 'b': pr[1]})
            @ init_state
        )
        ref_g = []
        for about_what in ('a', 'b'):
            ref_g.append(
                (
                    init_state.T.conj()
                    @ circ.hermitian().matrix({'a': pr[0], 'b': pr[1]})
                    @ ham0.hamiltonian.matrix(3)
                    @ circ1.matrix()
                    @ np.kron(np.eye(4, 4), pr_gate.diff_matrix({'a': pr[0], 'b': pr[1]}, about_what))
                    @ circ0.matrix()
                    @ init_state
                ).real
                * 2
            )
        assert np.allclose(f, ref_f, atol=1e-4)
        assert np.allclose(g, ref_g, atol=1e-4)


def three_qubits_dm_evolution_in_py(dm, g, dtype):
    """
    Description: test three qubits density matrix
    Expectation: success.
    """
    if isinstance(g, G.NoiseGate):
        tmp = np.zeros((8, 8), dtype=mq.to_np_type(dtype))
        for m in g.matrix():
            if g.obj_qubits[0] == 0:
                big_m = np.kron(np.eye(4, 4), m)
            elif g.obj_qubits[0] == 1:
                big_m = np.kron(np.kron(np.eye(2, 2), m), np.eye(2, 2))
            else:
                big_m = np.kron(m, np.eye(4, 4))
            tmp += big_m @ dm @ big_m.conj().T
        dm = tmp
    else:
        if g.obj_qubits[0] == 0:
            big_m = np.kron(np.eye(4, 4), g.matrix())
        elif g.obj_qubits[0] == 1:
            big_m = np.kron(np.kron(np.eye(2, 2), g.matrix()), np.eye(2, 2))
        else:
            big_m = np.kron(g.matrix(), np.eye(4, 4))
        dm = big_m @ dm @ big_m.conj().T
    return dm


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", ['mqmatrix'])
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_noise_get_expectation_with_grad(virtual_qc, dtype):
    """
    Description: test noise circuit get expectation with gradient
    Expectation: success.
    """
    # pylint: disable=too-many-locals
    init_state = np.random.rand(8) + np.random.rand(8) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    init_dm = np.outer(init_state, init_state.conj())
    circ0 = random_circuit(3, 100, 1.0, 0.0)
    circ1 = random_circuit(3, 100, 1.0, 0.0)
    circ = circ0 + G.RX({'a': 1, 'b': 2}).on(0) + circ1
    circ = circ.with_noise()
    ham0 = Hamiltonian(QubitOperator('X0 Y1') + QubitOperator('Z0'), dtype=dtype)
    ham1 = ham0.sparse(3)
    ham2 = Hamiltonian(csr_matrix(ham0.hamiltonian.matrix(3)), dtype=dtype)
    for ham in (ham0, ham1, ham2):
        sim = Simulator(virtual_qc, 3, dtype=dtype)
        sim.set_qs(init_dm)
        grad_ops = sim.get_expectation_with_grad(ham, circ)
        pr = np.random.rand(2) * 2 * np.pi
        f, grad = grad_ops(pr)
        sim.apply_circuit(circ, pr)
        dm = sim.get_qs()
        ref_f = np.trace(ham0.hamiltonian.matrix(3) @ dm)
        ref_grad = []
        for about_what in ('a', 'b'):
            dm = init_dm
            for g in circ:
                if g.parameterized:
                    dm = (
                        np.kron(np.eye(4, 4), g.diff_matrix({'a': pr[0], 'b': pr[1]}, about_what))
                        @ dm
                        @ np.kron(np.eye(4, 4), g.hermitian().matrix({'a': pr[0], 'b': pr[1]}))
                    )
                else:
                    dm = three_qubits_dm_evolution_in_py(dm, g, dtype)
            ref_grad.append(np.trace(ham0.hamiltonian.matrix(3) @ dm).real * 2)
        assert np.allclose(f, ref_f, atol=1e-6)
        assert np.allclose(grad, ref_grad, atol=1e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_entropy(config):
    """
    Description: test entropy
    Expectation: success.
    """
    virtual_qc, dtype = config
    circ = random_circuit(3, 100)
    circ = circ.with_noise(G.DepolarizingChannel(0.1))
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    sim.apply_circuit(circ)
    e = sim.entropy()
    if virtual_qc.startswith('mqvector'):
        ref_entropy = 0
    else:
        dm = sim.get_qs()
        ref_entropy = -np.trace(dm @ logm(dm))
    assert np.allclose(e, ref_entropy, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config1", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.parametrize("config2", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_fidelity(config1, config2):
    """
    Description: test fidelity of two quantum states
    Expectation: success.
    """
    virtual_qc1, dtype1 = config1
    virtual_qc2, dtype2 = config2
    circ1 = random_circuit(3, 100)
    circ2 = random_circuit(3, 100)
    sim1 = Simulator(virtual_qc1, 3, dtype=dtype1)
    sim2 = Simulator(virtual_qc2, 3, dtype=dtype2)
    sim1.apply_circuit(circ1)
    sim2.apply_circuit(circ2)
    qs1 = sim1.get_qs()
    qs2 = sim2.get_qs()
    f = fidelity(qs1, qs2)
    if virtual_qc1.startswith('mqvector'):
        qs1 = np.outer(qs1, qs1.conj().T)
    if virtual_qc2.startswith('mqvector'):
        qs2 = np.outer(qs2, qs2.conj().T)
    ref_f = np.trace(sqrtm(sqrtm(qs1) @ qs2 @ sqrtm(qs1))).real ** 2
    assert np.allclose(f, ref_f, atol=1e-3)
