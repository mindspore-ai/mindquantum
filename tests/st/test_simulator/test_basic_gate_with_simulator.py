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
"""Test basic gate with simulator."""
# pylint: disable=invalid-unary-operand-type
from inspect import signature

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR
from mindquantum.utils import random_circuit

none_parameter_gate = [
    G.HGate,
    G.IGate,
    G.ISWAPGate,
    G.SGate,
    G.SWAPGate,
    G.TGate,
    G.XGate,
    G.YGate,
    G.ZGate,
    G.SXGate,
    lambda: G.SXGate().hermitian(),
    lambda: G.TGate().hermitian(),
    lambda: G.SGate().hermitian(),
]

single_parameter_gate = [
    G.RX,
    G.RY,
    G.RZ,
    G.SWAPalpha,
    G.Givens,
    G.GlobalPhase,
    G.PhaseShift,
    G.Rxx,
    G.Rxy,
    G.Rxz,
    G.Ryy,
    G.Ryz,
    G.Rzz,
]

multi_parameter_gate = [
    G.U3,
    G.FSim,
    G.Rn,
]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.parametrize("gate", none_parameter_gate)
def test_none_parameter_gate(config, gate):
    """
    Description: test none parameter gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    g = gate()
    dim = 2**g.n_qubits
    g = g.on(list(range(g.n_qubits)))
    init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
    sim = Simulator(virtual_qc, g.n_qubits, dtype=dtype)
    sim.set_qs(init_state)
    sim.apply_gate(g)
    ref_qs = g.matrix() @ (init_state / np.linalg.norm(init_state))
    if virtual_qc.startswith("mqmatrix"):
        assert np.allclose(sim.get_qs(), np.outer(ref_qs, ref_qs.conj()))
    else:
        assert np.allclose(sim.get_qs(), ref_qs)

    c_g = g.on(list(range(g.n_qubits)), g.n_qubits)
    c_init_state = np.random.rand(2 * dim) + np.random.rand(2 * dim) * 1j
    c_sim = Simulator(virtual_qc, c_g.n_qubits + 1, dtype=dtype)
    c_sim.set_qs(c_init_state)
    c_sim.apply_gate(c_g)
    m = np.block([[np.eye(dim), np.zeros((dim, dim))], [np.zeros((dim, dim)), g.matrix()]])
    c_ref_qs = m @ (c_init_state / np.linalg.norm(c_init_state))
    if virtual_qc.startswith("mqmatrix"):
        assert np.allclose(c_sim.get_qs(), np.outer(c_ref_qs, c_ref_qs.conj()), atol=1e-6)
    else:
        assert np.allclose(c_sim.get_qs(), c_ref_qs, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.parametrize("gate", single_parameter_gate + multi_parameter_gate)
def test_parameter_gate(config, gate):  # pylint: disable=too-many-locals
    """
    Description: test all parameter gates
    Expectation: success.
    """
    virtual_qc, dtype = config
    n_pr = len(signature(gate).parameters)
    pr = np.random.rand(n_pr) * 2 * np.pi
    g = gate(*pr)
    dim = 2**g.n_qubits
    g = g.on(list(range(g.n_qubits)))
    init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
    sim = Simulator(virtual_qc, g.n_qubits, dtype=dtype)
    sim.set_qs(init_state)
    sim.apply_gate(g)
    ref_qs = g.matrix() @ (init_state / np.linalg.norm(init_state))
    if virtual_qc.startswith("mqmatrix"):
        assert np.allclose(sim.get_qs(), np.outer(ref_qs, ref_qs.conj()), atol=1e-6)
    else:
        assert np.allclose(sim.get_qs(), ref_qs, atol=1e-6)

    c_g = g.on(list(range(g.n_qubits)), g.n_qubits)
    c_init_state = np.random.rand(2 * dim) + np.random.rand(2 * dim) * 1j
    c_sim = Simulator(virtual_qc, c_g.n_qubits + 1, dtype=dtype)
    c_sim.set_qs(c_init_state)
    c_sim.apply_gate(c_g)
    m = np.block([[np.eye(dim), np.zeros((dim, dim))], [np.zeros((dim, dim)), g.matrix()]])
    c_ref_qs = m @ (c_init_state / np.linalg.norm(c_init_state))
    if virtual_qc.startswith("mqmatrix"):
        assert np.allclose(c_sim.get_qs(), np.outer(c_ref_qs, c_ref_qs.conj()), atol=1e-6)
    else:
        assert np.allclose(c_sim.get_qs(), c_ref_qs, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.parametrize("gate", single_parameter_gate)
def test_single_parameter_gate_expectation_with_grad(config, gate):  # pylint: disable=R0914
    """
    Description: test expectation and gradient of single parameter gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    g = gate({'a': 1, 'b': 2})
    dim = 2**g.n_qubits
    g = g.on(list(range(g.n_qubits)))
    init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    ham = Hamiltonian(QubitOperator('X0') + QubitOperator('Z0'), dtype=dtype)
    sim = Simulator(virtual_qc, g.n_qubits, dtype=dtype)
    sim.set_qs(init_state)
    grad_ops = sim.get_expectation_with_grad(ham, Circuit(g))
    pr = np.random.rand(2) * 2 * np.pi
    f, grad = grad_ops(pr)
    ref_f = (
        init_state.T.conj()
        @ g.hermitian().matrix({'a': pr[0], 'b': pr[1]})
        @ ham.hamiltonian.matrix(g.n_qubits)
        @ g.matrix({'a': pr[0], 'b': pr[1]})
        @ init_state
    )
    ref_grad = []
    for about_what in ('a', 'b'):
        ref_grad.append(
            (
                init_state.T.conj()
                @ g.hermitian().matrix({'a': pr[0], 'b': pr[1]})
                @ ham.hamiltonian.matrix(g.n_qubits)
                @ g.diff_matrix({'a': pr[0], 'b': pr[1]}, about_what)
                @ init_state
            ).real
            * 2
        )
    assert np.allclose(f, ref_f, atol=1e-4)
    assert np.allclose(grad, ref_grad, atol=1e-4)

    c_g = g.on(list(range(g.n_qubits)), g.n_qubits)
    c_init_state = np.random.rand(2 * dim) + np.random.rand(2 * dim) * 1j
    c_init_state = c_init_state / np.linalg.norm(c_init_state)
    c_sim = Simulator(virtual_qc, c_g.n_qubits + 1, dtype=dtype)
    c_sim.set_qs(c_init_state)
    c_grad_ops = c_sim.get_expectation_with_grad(ham, Circuit(c_g))
    c_pr = np.random.rand(2) * 2 * np.pi
    c_f, c_grad = c_grad_ops(c_pr)
    m = np.block([[np.eye(dim), np.zeros((dim, dim))], [np.zeros((dim, dim)), g.matrix({'a': c_pr[0], 'b': c_pr[1]})]])
    c_ref_f = c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(g.n_qubits + 1) @ m @ c_init_state
    c_ref_grad = []
    for about_what in ('a', 'b'):
        diff_m = np.block(
            [
                [np.zeros((dim, dim)), np.zeros((dim, dim))],
                [np.zeros((dim, dim)), g.diff_matrix({'a': c_pr[0], 'b': c_pr[1]}, about_what)],
            ]
        )
        c_ref_grad.append(
            2
            * (c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(g.n_qubits + 1) @ diff_m @ c_init_state).real
        )
    assert np.allclose(c_f, c_ref_f, atol=1e-5)
    assert np.allclose(c_grad, c_ref_grad, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_custom_gate(config):  # pylint: disable=too-many-locals
    """
    Description: test custom gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    for n in (1, 2, 3):
        circ = mq.random_circuit(n, 100)
        g = G.UnivMathGate('custom', circ.matrix())
        dim = 2 ** (n + 1)
        g_dim = 2**n
        g = g.on(list(range(g.n_qubits)))
        init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
        sim = Simulator(virtual_qc, n + 1, dtype=dtype)
        sim.set_qs(init_state)
        sim.apply_gate(g)
        ref_qs = np.kron(np.eye(2), g.matrix()) @ (init_state / np.linalg.norm(init_state))
        if virtual_qc.startswith("mqmatrix"):
            assert np.allclose(sim.get_qs(), np.outer(ref_qs, ref_qs.conj()), atol=1e-6)
        else:
            assert np.allclose(sim.get_qs(), ref_qs, atol=1e-6)

        c_g = g.on(list(range(g.n_qubits)), g.n_qubits)
        c_init_state = np.random.rand(2 * dim) + np.random.rand(2 * dim) * 1j
        c_sim = Simulator(virtual_qc, n + 2, dtype=dtype)
        c_sim.set_qs(c_init_state)
        c_sim.apply_gate(c_g)
        m = np.block([[np.eye(g_dim), np.zeros((g_dim, g_dim))], [np.zeros((g_dim, g_dim)), g.matrix()]])
        c_ref_qs = np.kron(np.eye(2), m) @ (c_init_state / np.linalg.norm(c_init_state))
        if virtual_qc.startswith("mqmatrix"):
            assert np.allclose(c_sim.get_qs(), np.outer(c_ref_qs, c_ref_qs.conj()), atol=1e-6)
        else:
            assert np.allclose(c_sim.get_qs(), c_ref_qs, atol=1e-6)


try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

try:
    importlib_metadata.import_module("numba")
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_custom_gate_expectation_with_grad(config):  # pylint: disable=too-many-locals
    """
    Description: test custom gate expectation with grad
    Expectation: success.
    """
    virtual_qc, dtype = config
    for n in (1, 2, 3):
        if n == 1:

            def matrix(alpha):
                ep = 0.5 * (1 + np.exp(1j * np.pi * alpha))
                em = 0.5 * (1 - np.exp(1j * np.pi * alpha))
                return np.array([[ep, em], [em, ep]])

            def diff_matrix(alpha):
                ep = 0.5 * 1j * np.pi * np.exp(1j * np.pi * alpha)
                em = -0.5 * 1j * np.pi * np.exp(1j * np.pi * alpha)
                return np.array([[ep, em], [em, ep]])

        elif n == 2:

            def matrix(alpha):
                ep = 0.5 * (1 + np.exp(1j * np.pi * alpha))
                em = 0.5 * (1 - np.exp(1j * np.pi * alpha))
                return np.array(
                    [
                        [1.0 + 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, ep, em, 0.0j],
                        [0.0j, em, ep, 0.0j],
                        [0.0j, 0.0j, 0.0j, 1.0 + 0.0j],
                    ]
                )

            def diff_matrix(alpha):
                ep = 0.5 * 1j * np.pi * np.exp(1j * np.pi * alpha)
                em = -0.5 * 1j * np.pi * np.exp(1j * np.pi * alpha)
                return np.array(
                    [
                        [0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, ep, em, 0.0j],
                        [0.0j, em, ep, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j],
                    ]
                )

        else:

            def matrix(alpha):
                ep = 0.5 * (1 + np.exp(1j * np.pi * alpha))
                em = 0.5 * (1 - np.exp(1j * np.pi * alpha))
                return np.array(
                    [
                        [1.0 + 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, 1.0 + 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 1.0 + 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, ep, em, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, em, ep, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 1.0 + 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 1.0 + 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 1.0 + 0.0j],
                    ]
                )

            def diff_matrix(alpha):
                ep = 0.5 * 1j * np.pi * np.exp(1j * np.pi * alpha)
                em = -0.5 * 1j * np.pi * np.exp(1j * np.pi * alpha)
                return np.array(
                    [
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, ep, em, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, em, ep, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                        [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j],
                    ]
                )

        g = G.gene_univ_parameterized_gate('custom', matrix, diff_matrix)
        dim = 2 ** (n + 1)
        g_dim = 2**n
        g = g('a').on(list(range(n)))
        init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
        init_state = init_state / np.linalg.norm(init_state)
        ham = Hamiltonian(QubitOperator('X0') + QubitOperator('Z0'), dtype=dtype)
        sim = Simulator(virtual_qc, n + 1, dtype=dtype)
        sim.set_qs(init_state)
        grad_ops = sim.get_expectation_with_grad(ham, Circuit(g))
        pr = np.random.rand() * 2 * np.pi
        f, grad = grad_ops([pr])
        ref_f = (
            init_state.T.conj()
            @ np.kron(np.eye(2), g.hermitian().matrix({'a': pr}))
            @ ham.hamiltonian.matrix(n + 1)
            @ np.kron(np.eye(2), g.matrix({'a': pr}))
            @ init_state
        )
        ref_grad = (
            init_state.T.conj()
            @ np.kron(np.eye(2), g.hermitian().matrix({'a': pr}))
            @ ham.hamiltonian.matrix(n + 1)
            @ np.kron(np.eye(2), g.diff_matrix({'a': pr}))
            @ init_state
        ).real * 2
        assert np.allclose(f, ref_f, atol=1e-6)
        assert np.allclose(grad, ref_grad, atol=1e-6)

        c_g = g.on(list(range(n)), n)
        c_init_state = np.random.rand(2 * dim) + np.random.rand(2 * dim) * 1j
        c_init_state = c_init_state / np.linalg.norm(c_init_state)
        c_sim = Simulator(virtual_qc, n + 2, dtype=dtype)
        c_sim.set_qs(c_init_state)
        c_grad_ops = c_sim.get_expectation_with_grad(ham, Circuit(c_g))
        c_pr = np.random.rand() * 2 * np.pi
        c_f, c_grad = c_grad_ops([c_pr])
        m = np.kron(
            np.eye(2),
            np.block([[np.eye(g_dim), np.zeros((g_dim, g_dim))], [np.zeros((g_dim, g_dim)), g.matrix({'a': c_pr})]]),
        )
        diff_m = np.kron(
            np.eye(2),
            np.block(
                [
                    [np.zeros((g_dim, g_dim)), np.zeros((g_dim, g_dim))],
                    [np.zeros((g_dim, g_dim)), g.diff_matrix({'a': c_pr})],
                ]
            ),
        )
        c_ref_f = c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(n + 2) @ m @ c_init_state
        c_ref_grad = (
            2 * (c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(n + 2) @ diff_m @ c_init_state).real
        )
        assert np.allclose(c_f, c_ref_f, atol=1e-6)
        assert np.allclose(c_grad, c_ref_grad, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_u3_expectation_with_grad(config):  # pylint: disable=R0914
    """
    Description: test expectation and gradient of U3 gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    g = G.U3('theta', 'phi', 'lamda')
    ref_g = [G.RZ('phi'), G.RX(-np.pi / 2), G.RZ('theta'), G.RX(np.pi / 2), G.RZ('lamda')]
    ref_g.reverse()
    dim = 2**g.n_qubits
    g = g.on(list(range(g.n_qubits)))
    ref_g = [i.on(list(range(g.n_qubits))) for i in ref_g]
    init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    ham = Hamiltonian(QubitOperator('X0') + QubitOperator('Z0'), dtype=dtype)
    sim = Simulator(virtual_qc, g.n_qubits, dtype=dtype)
    sim.set_qs(init_state)
    grad_ops = sim.get_expectation_with_grad(ham, Circuit(g))
    ref_grad_ops = sim.get_expectation_with_grad(ham, Circuit(ref_g))
    theta, phi, lamda = np.random.rand(3) * 2 * np.pi
    pr = [theta, phi, lamda]
    ref_pr = [lamda, theta, phi]
    f, grad = grad_ops(pr)
    ref_f, ref_grad = ref_grad_ops(ref_pr)
    ref_grad = np.array([ref_grad[0][0][1], ref_grad[0][0][2], ref_grad[0][0][0]])
    assert np.allclose(f, ref_f, atol=1e-6)
    assert np.allclose(grad, ref_grad.real, atol=1e-6)

    c_g = g.on(list(range(g.n_qubits)), g.n_qubits)
    ref_c_g = [i.on(list(range(g.n_qubits)), g.n_qubits) for i in ref_g]
    c_init_state = np.random.rand(2 * dim) + np.random.rand(2 * dim) * 1j
    c_init_state = c_init_state / np.linalg.norm(c_init_state)
    c_sim = Simulator(virtual_qc, c_g.n_qubits + 1, dtype=dtype)
    c_sim.set_qs(c_init_state)
    c_grad_ops = c_sim.get_expectation_with_grad(ham, Circuit(c_g))
    ref_c_grad_ops = c_sim.get_expectation_with_grad(ham, Circuit(ref_c_g))
    theta, phi, lamda = np.random.rand(3) * 2 * np.pi
    c_pr = [theta, phi, lamda]
    ref_c_pr = [lamda, theta, phi]
    c_f, c_grad = c_grad_ops(c_pr)
    ref_c_f, ref_c_grad = ref_c_grad_ops(ref_c_pr)
    ref_c_grad = np.array([ref_c_grad[0][0][1], ref_c_grad[0][0][2], ref_c_grad[0][0][0]])
    assert np.allclose(c_f, ref_c_f, atol=1e-6)
    assert np.allclose(c_grad, ref_c_grad.real, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_fsim_expectation_with_grad(config):  # pylint: disable=R0914
    """
    Description: test expectation and gradient of FSim gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    g = G.FSim('theta', 'phi')
    dim = 2**g.n_qubits

    def theta_diff_matrix(theta):
        return np.array(
            [
                [0, 0, 0, 0],
                [0, -np.sin(theta), -1j * np.cos(theta), 0],
                [0, -1j * np.cos(theta), -np.sin(theta), 0],
                [0, 0, 0, 0],
            ]
        )

    def phi_diff_matrix(phi):
        return np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, -1j * np.exp(-1j * phi)],
            ]
        )

    g = g.on(list(range(g.n_qubits)))
    init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    ham = Hamiltonian(QubitOperator('X0') + QubitOperator('Z0'), dtype=dtype)
    sim = Simulator(virtual_qc, g.n_qubits, dtype=dtype)
    sim.set_qs(init_state)
    grad_ops = sim.get_expectation_with_grad(ham, Circuit(g))
    pr = np.random.rand(2) * 2 * np.pi
    f, grad = grad_ops(pr)
    ref_f = (
        init_state.T.conj()
        @ g.hermitian().matrix({'theta': pr[0], 'phi': pr[1]})
        @ ham.hamiltonian.matrix(g.n_qubits)
        @ g.matrix({'theta': pr[0], 'phi': pr[1]})
        @ init_state
    )
    ref_grad_theta = (
        init_state.T.conj()
        @ g.hermitian().matrix({'theta': pr[0], 'phi': pr[1]})
        @ ham.hamiltonian.matrix(g.n_qubits)
        @ theta_diff_matrix(pr[0])
        @ init_state
    ).real * 2
    ref_grad_phi = (
        init_state.T.conj()
        @ g.hermitian().matrix({'theta': pr[0], 'phi': pr[1]})
        @ ham.hamiltonian.matrix(g.n_qubits)
        @ phi_diff_matrix(pr[1])
        @ init_state
    ).real * 2
    ref_grad = np.array([ref_grad_theta, ref_grad_phi])
    assert np.allclose(f, ref_f, atol=1e-6)
    assert np.allclose(grad, ref_grad, atol=1e-6)

    c_g = g.on(list(range(g.n_qubits)), g.n_qubits)
    c_init_state = np.random.rand(2 * dim) + np.random.rand(2 * dim) * 1j
    c_init_state = c_init_state / np.linalg.norm(c_init_state)
    c_sim = Simulator(virtual_qc, c_g.n_qubits + 1, dtype=dtype)
    c_sim.set_qs(c_init_state)
    c_grad_ops = c_sim.get_expectation_with_grad(ham, Circuit(c_g))
    c_pr = np.random.rand(2) * 2 * np.pi
    c_f, c_grad = c_grad_ops(c_pr)
    m = np.block(
        [[np.eye(dim), np.zeros((dim, dim))], [np.zeros((dim, dim)), g.matrix({'theta': c_pr[0], 'phi': c_pr[1]})]]
    )
    diff_m_theta = np.block(
        [[np.zeros((dim, dim)), np.zeros((dim, dim))], [np.zeros((dim, dim)), theta_diff_matrix(c_pr[0])]]
    )
    diff_m_phi = np.block(
        [[np.zeros((dim, dim)), np.zeros((dim, dim))], [np.zeros((dim, dim)), phi_diff_matrix(c_pr[1])]]
    )
    c_ref_f = c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(g.n_qubits + 1) @ m @ c_init_state
    c_ref_grad_theta = (
        2
        * (
            c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(g.n_qubits + 1) @ diff_m_theta @ c_init_state
        ).real
    )
    c_ref_grad_phi = (
        2
        * (c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(g.n_qubits + 1) @ diff_m_phi @ c_init_state).real
    )
    c_ref_grad = np.array([c_ref_grad_theta, c_ref_grad_phi])
    assert np.allclose(c_f, c_ref_f, atol=1e-6)
    assert np.allclose(c_grad, c_ref_grad, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_rn_expectation_with_grad(config):  # pylint: disable=R0914
    """
    Description: test expectation and gradient of Rn gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    rn = G.Rn('a', 'b', 'c').on(0)
    init = mq.random_circuit(1, 10)
    circ = Circuit([rn])
    ham = Hamiltonian(QubitOperator('X0') + QubitOperator('Y0') + QubitOperator('Z0')).astype(dtype)
    m_ham = ham.hamiltonian.matrix().toarray()
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    grad_ops = sim.get_expectation_with_grad(ham, init + circ)
    p0 = np.random.uniform(-3, 3, 3)
    f, g = grad_ops(p0)
    f = f[0, 0]
    g = g[0, 0]
    m_rn = rn.matrix(pr=dict(zip(circ.params_name, p0)))
    psi_0 = init.get_qs()[:, None]
    f_exp = np.vdot(m_rn @ psi_0, m_ham @ m_rn @ psi_0)
    assert np.allclose(f, f_exp, atol=1e-4)
    delta = 0.0001
    f1, _ = grad_ops(p0 + np.array([delta, 0, 0]))
    f1 = f1[0, 0]
    f2, _ = grad_ops(p0 + np.array([0, delta, 0]))
    f2 = f2[0, 0]
    f3, _ = grad_ops(p0 + np.array([0, 0, delta]))
    f3 = f3[0, 0]
    g_exp = (np.array([f1, f2, f3]) - f) / delta
    assert np.allclose(g, g_exp, atol=0.01)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_pauli_string_gate(config):  # pylint: disable=too-many-locals
    """
    Description: test pauli string gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    n_qubits = 5

    def random_pauli_string(obj_qubits):
        pauli = ['X', 'Y', 'Z']
        return ''.join(i for i in np.random.choice(pauli, len(obj_qubits)))

    circs = []
    for i in range(20):
        qubits = list(range(n_qubits))
        np.random.shuffle(qubits)
        obj_qubits = qubits[: n_qubits - 1]
        ctrl_qubits = None if np.random.random() < 0.5 else qubits[-1]
        circs.append(G.GroupedPauli(random_pauli_string(obj_qubits)).on(obj_qubits, ctrl_qubits))
    if virtual_qc.startswith('mqvector'):
        state = (random_circuit(n_qubits, 10) + G.I.on(n_qubits - 1)).get_qs()
        sim = Simulator(virtual_qc, n_qubits, dtype=dtype)
        for g in circs:
            sim.set_qs(state)
            sim.apply_gate(g)
            qs0 = sim.get_qs()
            sim.set_qs(state)
            sim.apply_circuit(g.__decompose__())
            qs1 = sim.get_qs()
            assert np.allclose(qs0, qs1)
    elif virtual_qc.startswith("mqmatrix"):
        qs0 = (random_circuit(n_qubits, 10) + G.I.on(n_qubits - 1)).get_qs()
        qs1 = (random_circuit(n_qubits, 10) + G.I.on(n_qubits - 1)).get_qs()
        qs2 = (random_circuit(n_qubits, 10) + G.I.on(n_qubits - 1)).get_qs()
        qs0 = np.outer(qs0, np.conj(qs0))
        qs1 = np.outer(qs1, np.conj(qs1))
        qs2 = np.outer(qs2, np.conj(qs2))
        state = 0.2 * qs0 + 0.3 * qs1 + 0.5 * qs2
        sim = Simulator(virtual_qc, n_qubits, dtype=dtype)
        for g in circs:
            sim.set_qs(state)
            sim.apply_gate(g)
            qs0 = sim.get_qs()
            sim.set_qs(state)
            sim.apply_circuit(g.__decompose__())
            qs1 = sim.get_qs()
            assert np.allclose(qs0, qs1)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_rot_pauli_string_gate(config):  # pylint: disable=too-many-locals
    """
    Description: test pauli string gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    n_qubits = 5
    for _ in range(20):
        qubits = list(range(n_qubits))
        np.random.shuffle(qubits)
        n_obj = np.random.randint(1, n_qubits + 1)
        if n_obj == n_qubits:
            n_ctrl = None
        else:
            n_ctrl = np.random.randint(0, n_qubits - n_obj)
        obj_qubits = qubits[:n_obj]
        if n_ctrl is None or n_ctrl == 0:
            ctrl_qubits = None
        else:
            ctrl_qubits = qubits[-n_ctrl:]
        pauli_string = ''.join(np.random.choice(['X', 'Y', 'Z'], n_obj))
        p0 = np.random.uniform(-3, 3)
        g = G.RotPauliString(pauli_string, p0).on(obj_qubits, ctrl_qubits)
        circ = g.__decompose__()
        sim = Simulator(virtual_qc, n_qubits, dtype=dtype)
        rand_circ = random_circuit(n_qubits, 20)
        sim.apply_circuit(rand_circ)
        sim.apply_gate(g)
        qs1 = sim.get_qs()
        sim.reset()
        sim.apply_circuit(rand_circ)
        sim.apply_circuit(circ)
        qs2 = sim.get_qs()
        if dtype == mq.complex64:
            atol = 1e-4
        else:
            atol = 1e-8
        assert np.allclose(qs1, qs2, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_rot_pauli_string_gate_gradient(config):  # pylint: disable=too-many-locals
    """
    Description: test pauli string gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    n_qubits = 5
    for _ in range(20):
        qubits = list(range(n_qubits))
        np.random.shuffle(qubits)
        n_obj = np.random.randint(1, n_qubits + 1)
        if n_obj == n_qubits:
            n_ctrl = None
        else:
            n_ctrl = np.random.randint(0, n_qubits - n_obj)
        obj_qubits = qubits[:n_obj]
        if n_ctrl is None or n_ctrl == 0:
            ctrl_qubits = None
        else:
            ctrl_qubits = qubits[-n_ctrl:]
        pauli_string = ''.join(np.random.choice(['X', 'Y', 'Z'], n_obj))
        p0 = np.random.uniform(-3, 3, size=(1,))
        g = G.RotPauliString(pauli_string, 'a').on(obj_qubits, ctrl_qubits)
        circ = g.__decompose__()
        ham = Hamiltonian(
            QubitOperator(
                ' '.join(f"{p}{idx}" for idx, p in enumerate(np.random.choice(['X', 'Y', 'Z'], n_qubits)))
            ).astype(dtype)
        )
        sim1 = Simulator(virtual_qc, n_qubits, dtype=dtype)
        sim2 = Simulator(virtual_qc, n_qubits, dtype=dtype)
        rand_circ = random_circuit(n_qubits, 20)
        rc2 = random_circuit(n_qubits, 20)
        circ1 = rand_circ + g + rc2
        circ2 = rand_circ + circ + rc2
        grad_ops1 = sim1.get_expectation_with_grad(ham, circ1)
        grad_ops2 = sim2.get_expectation_with_grad(ham, circ2)
        f1, g1 = grad_ops1(p0)
        f2, g2 = grad_ops2(p0)
        if dtype == mq.complex64:
            atol = 1e-4
        else:
            atol = 1e-8
        assert np.allclose(f1, f2, atol=atol)
        assert np.allclose(g1, g2, atol=atol)
