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
from inspect import signature

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR

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
]

single_parameter_gate = [
    G.RX,
    G.RY,
    G.RZ,
    G.SWAPalpha,
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

multi_parameter_gate = [
    G.U3,
    G.FSim,
]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(SUPPORTED_SIMULATOR))
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
@pytest.mark.parametrize("config", list(SUPPORTED_SIMULATOR))
@pytest.mark.parametrize("gate", single_parameter_gate + multi_parameter_gate)
def test_parameter_gate(config, gate):
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
@pytest.mark.parametrize("config", list(SUPPORTED_SIMULATOR))
@pytest.mark.parametrize("gate", single_parameter_gate)
def test_single_parameter_gate_expectation_with_grad(config, gate):  # pylint: disable=R0914
    """
    Description: test expectation and gradient of single parameter gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    g = gate('a')
    dim = 2**g.n_qubits
    g = g.on(list(range(g.n_qubits)))
    init_state = np.random.rand(dim) + np.random.rand(dim) * 1j
    init_state = init_state / np.linalg.norm(init_state)
    ham = Hamiltonian(QubitOperator('X0') + QubitOperator('Z0'), dtype=dtype)
    sim = Simulator(virtual_qc, g.n_qubits, dtype=dtype)
    sim.set_qs(init_state)
    grad_ops = sim.get_expectation_with_grad(ham, Circuit(g))
    pr = np.random.rand() * 2 * np.pi
    f, grad = grad_ops([pr])
    ref_f = (
        init_state.T.conj()
        @ g.hermitian().matrix({'a': pr})
        @ ham.hamiltonian.matrix(g.n_qubits)
        @ g.matrix({'a': pr})
        @ init_state
    )
    ref_grad = (
        init_state.T.conj()
        @ g.hermitian().matrix({'a': pr})
        @ ham.hamiltonian.matrix(g.n_qubits)
        @ g.diff_matrix({'a': pr})
        @ init_state
    ).real * 2
    assert np.allclose(f, ref_f, atol=1e-6)
    assert np.allclose(grad, ref_grad.real, atol=1e-4)

    c_g = g.on(list(range(g.n_qubits)), g.n_qubits)
    c_init_state = np.random.rand(2 * dim) + np.random.rand(2 * dim) * 1j
    c_init_state = c_init_state / np.linalg.norm(c_init_state)
    c_sim = Simulator(virtual_qc, c_g.n_qubits + 1, dtype=dtype)
    c_sim.set_qs(c_init_state)
    c_grad_ops = c_sim.get_expectation_with_grad(ham, Circuit(c_g))
    c_pr = np.random.rand() * 2 * np.pi
    c_f, c_grad = c_grad_ops([c_pr])
    m = np.block([[np.eye(dim), np.zeros((dim, dim))], [np.zeros((dim, dim)), g.matrix({'a': c_pr})]])
    diff_m = np.block(
        [[np.zeros((dim, dim)), np.zeros((dim, dim))], [np.zeros((dim, dim)), g.diff_matrix({'a': c_pr})]]
    )
    c_ref_f = c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(g.n_qubits + 1) @ m @ c_init_state
    c_ref_grad = (
        2 * (c_init_state.T.conj() @ m.T.conj() @ ham.hamiltonian.matrix(g.n_qubits + 1) @ diff_m @ c_init_state).real
    )
    assert np.allclose(c_f, c_ref_f, atol=1e-6)
    assert np.allclose(c_grad, c_ref_grad, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(SUPPORTED_SIMULATOR))
def test_custom_gate(config):
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
@pytest.mark.parametrize("config", list(SUPPORTED_SIMULATOR))
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_custom_gate_expectation_with_grad(config):
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
        assert np.allclose(grad, ref_grad.real, atol=1e-6)

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
