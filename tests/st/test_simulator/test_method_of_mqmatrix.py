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

import itertools
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
from mindquantum.utils import random_circuit, random_hamiltonian


@pytest.mark.level0
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
    ham0 = random_hamiltonian(3, 10, dtype=dtype)
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
    ham0 = random_hamiltonian(3, 10, dtype=dtype)
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_get_expectation_with_grad_batch_hams(config):
    """
    Description: test get expectation with gradient with batch hams
    Expectation: success.
    """
    # pylint: disable=too-many-locals
    virtual_qc, dtype = config
    init_state_circ = UN(G.H, range(5))
    ansatz = mq.MaxCutAnsatz([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 2)], 4).circuit
    circ = init_state_circ + ansatz
    sim = Simulator(virtual_qc, 5, dtype=dtype)

    def numbers_to_binary_matrix(n):
        max_bits = int(np.ceil(np.log2(n)))
        binary_matrix = np.zeros((n, max_bits), dtype=int)
        for i in range(n):
            binary_representation = list(bin(i)[2:])
            binary_matrix[i, -len(binary_representation) :] = list(map(int, binary_representation))
        return binary_matrix

    qubit_num = 5
    output_dim = 2**qubit_num
    max_bits = int(np.ceil(np.log2(output_dim)))
    hams = [None] * output_dim
    hams_sign = numbers_to_binary_matrix(output_dim)
    for index in range(output_dim):
        ham = QubitOperator('I0', 1)
        for jndex in range(max_bits):
            ham *= QubitOperator(f'I{jndex}', 0.5) + (-1) ** hams_sign[index, jndex] * QubitOperator(f'Z{jndex}', 0.5)
        hams[index] = Hamiltonian(ham, dtype=dtype)
    grad_ops = sim.get_expectation_with_grad(hams, circ)
    rng = np.random.default_rng(10)
    p0 = rng.random(size=len(circ.params_name)) * np.pi * 2 - np.pi
    f, g = grad_ops(p0)
    ref_f = []
    init_state = np.zeros(2**5)
    init_state[0] = 1
    for ham in hams:
        ref_f.append(
            init_state.T.conj()
            @ circ.hermitian().matrix(dict(zip(circ.params_name, p0)))
            @ ham.hamiltonian.matrix(5)
            @ circ.matrix(dict(zip(circ.params_name, p0)))
            @ init_state
        )
    assert np.allclose(f, ref_f, atol=1e-4)


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
    ham0 = random_hamiltonian(3, 10, dtype=dtype)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_get_reduced_density_matrix(config):
    """
    Description: test getting reduced density matrix functionality
    Expectation: success.
    """
    virtual_qc, dtype = config

    def partial_trace(rho, keep_qubits, n_qubits):
        """Implementation of partial trace"""
        trace_qubits = sorted(list(set(range(n_qubits)) - set(keep_qubits)))
        keep_dim = 2 ** len(keep_qubits)
        result = np.zeros((keep_dim, keep_dim), dtype=rho.dtype)
        steps = [2**k for k in keep_qubits]
        for i in range(keep_dim):
            for j in range(keep_dim):
                i_bits = [(i >> b) & 1 for b in range(len(keep_qubits))]
                j_bits = [(j >> b) & 1 for b in range(len(keep_qubits))]

                row_idx = sum(bit * step for bit, step in zip(i_bits, steps))
                col_idx = sum(bit * step for bit, step in zip(j_bits, steps))

                block_size = 2 ** len(trace_qubits)

                for k in range(block_size):
                    k_bits = [(k >> b) & 1 for b in range(len(trace_qubits))]
                    row_offset = sum(bit * (2**q) for bit, q in zip(k_bits, trace_qubits))
                    col_offset = row_offset
                    result[i, j] += rho[row_idx + row_offset, col_idx + col_offset]

        return result

    # Test basic functionality
    sim = Simulator(virtual_qc, 3, dtype=dtype)

    # Test 1: Simple Bell state
    circ = Circuit([G.H.on(0), G.CNOT.on(1, 0)])
    sim.apply_circuit(circ)
    qs = sim.get_qs()
    # Convert to density matrix if state vector
    if len(qs.shape) == 1:
        full_dm = np.outer(qs, qs.conj())
    else:
        full_dm = qs

    # Verify single qubit reduction
    rdm_0 = sim.get_reduced_density_matrix([0])
    ref_rdm_0 = partial_trace(full_dm, [0], 3)
    assert np.allclose(rdm_0, ref_rdm_0, atol=1e-6)
    assert rdm_0.shape == (2, 2)
    assert np.allclose(np.trace(rdm_0), 1.0, atol=1e-6)

    # Verify two qubit reduction
    rdm_01 = sim.get_reduced_density_matrix([0, 1])
    ref_rdm_01 = partial_trace(full_dm, [0, 1], 3)
    assert np.allclose(rdm_01, ref_rdm_01, atol=1e-6)
    assert rdm_01.shape == (4, 4)
    assert np.allclose(np.trace(rdm_01), 1.0, atol=1e-6)

    # Test 2: Random circuit
    sim.reset()
    random_circ = random_circuit(3, 10)
    sim.apply_circuit(random_circ)
    qs = sim.get_qs()
    # Convert to density matrix if state vector
    if len(qs.shape) == 1:
        full_dm = np.outer(qs, qs.conj())
    else:
        full_dm = qs

    # Test all possible qubit combinations
    for n_qubits in range(1, 3):
        for qubits in itertools.combinations(range(3), n_qubits):
            rdm = sim.get_reduced_density_matrix(list(qubits))
            ref_rdm = partial_trace(full_dm, list(qubits), 3)
            assert np.allclose(rdm, ref_rdm, atol=1e-6)
            assert rdm.shape == (2 ** len(qubits), 2 ** len(qubits))
            assert np.allclose(np.trace(rdm), 1.0, atol=1e-6)
            # Verify reduced density matrix is Hermitian
            assert np.allclose(rdm, rdm.conj().T, atol=1e-6)
            # Verify reduced density matrix is positive semidefinite
            eigenvals = np.linalg.eigvalsh(rdm)
            assert np.all(eigenvals >= -1e-6)

    # Test error cases
    with pytest.raises(ValueError):
        sim.get_reduced_density_matrix([0, 0])  # Duplicate qubits
    with pytest.raises(ValueError):
        sim.get_reduced_density_matrix([0, 1, 2, 3])  # Out of range qubits
    with pytest.raises(ValueError):
        sim.get_reduced_density_matrix([3])  # Invalid qubit index


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_get_qs_of_qubits(config):
    """
    Description: test getting quantum state of specified qubits
    Expectation: success.
    """
    virtual_qc, dtype = config
    # Prepare a 3-qubit system
    sim = Simulator(virtual_qc, 3, dtype=dtype)

    # Prepare a pure state where some subsystems are mixed
    # |ψ⟩ = |1⟩(|00⟩ + |11⟩)/√2
    circ_pure = Circuit([G.H.on(0), G.CNOT.on(1, 0), G.X.on(2)])
    sim.apply_circuit(circ_pure)

    # Test reduced state of single qubit (should be mixed)
    state_0 = sim.get_qs_of_qubits(0)
    assert len(state_0.shape) == 2  # Should return density matrix
    assert state_0.shape == (2, 2)
    assert np.allclose(np.trace(state_0), 1.0, atol=1e-6)
    # Verify it's a mixed state
    purity = np.real(np.trace(state_0 @ state_0))
    assert np.allclose(purity, 0.5, atol=1e-6)

    # Test reduced state of two entangled qubits (should be pure)
    state_01 = sim.get_qs_of_qubits([0, 1])
    assert len(state_01.shape) == 1  # Should return state vector
    assert state_01.shape[0] == 4
    # Verify it's indeed Bell state (|00⟩ + |11⟩)/√2
    expected_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(np.abs(state_01), np.abs(expected_state), atol=1e-6)

    # Test mixed state case (only for mqmatrix)
    if not virtual_qc.startswith('mqvector'):
        circ_mixed = circ_pure.with_noise(G.DepolarizingChannel(0.1))
        sim.reset()
        sim.apply_circuit(circ_mixed)
        state_mixed = sim.get_qs_of_qubits([0, 1])
        assert len(state_mixed.shape) == 2
        assert state_mixed.shape == (4, 4)
        assert np.allclose(np.trace(state_mixed), 1.0, atol=1e-6)
        # Verify it's a mixed state
        purity = np.real(np.trace(state_mixed @ state_mixed))
        assert purity < 0.99  # Purity should be less than 1 due to noise

    # Test ket format output
    state_str = sim.get_qs_of_qubits([1, 2], ket=True)
    assert isinstance(state_str, str)
    assert "(mixed state)" in state_str
    assert "¦10⟩" in state_str
    assert "¦11⟩" in state_str

    # Test error cases
    with pytest.raises(TypeError):
        sim.get_qs_of_qubits("0")  # Wrong input type
    with pytest.raises(ValueError):
        sim.get_qs_of_qubits([3])  # Invalid qubit index
