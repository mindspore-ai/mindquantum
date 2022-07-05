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
"""Test simulator."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mindquantum.core.operators as ops
from mindquantum import Circuit, Hamiltonian
from mindquantum import ParameterResolver as PR
from mindquantum import QubitOperator
from mindquantum.algorithm import qft
from mindquantum.core import gates as G
from mindquantum.core.circuit import UN
from mindquantum.simulator import inner_product
from mindquantum.simulator.simulator import Simulator

_has_mindspore = True
try:
    import mindspore as ms

    from mindquantum.framework.layer import MQAnsatzOnlyLayer

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
except ImportError:
    _has_mindspore = False


def _test_init_reset(virtual_qc):
    """
    test
    Description:
    Expectation:
    """
    s1 = Simulator(virtual_qc, 2)
    circ = Circuit().h(0).h(1)
    v1 = s1.get_qs()
    s1.apply_circuit(circ)
    s1.reset()
    v3 = s1.get_qs()
    v = np.array([1, 0, 0, 0], dtype=np.complex128)
    assert np.allclose(v1, v)
    assert np.allclose(v1, v3)


def _test_apply_circuit_and_hermitian(virtual_qc):
    """
    test
    Description:
    Expectation:
    """
    sv0 = np.array([[1, 0], [0, 0]])
    sv1 = np.array([[0, 0], [0, 1]])
    circ = Circuit()
    circ.ry(1.2, 0).ry(3.4, 1)
    circ.h(0).h(1)
    circ.x(1, 0)
    circ.rx('a', 0).ry('b', 1)
    circ.zz('c', (0, 1)).z(1, 0)
    s1 = Simulator(virtual_qc, circ.n_qubits)
    pr = PR({'a': 1, 'b': 3, 'c': 5})
    s1.apply_circuit(circ, pr)
    v1 = s1.get_qs()
    m = np.kron(G.RY(3.4).matrix(), G.RY(1.2).matrix())
    m = np.kron(G.H.matrix(), G.H.matrix()) @ m
    m = (np.kron(G.I.matrix(), sv0) + np.kron(G.X.matrix(), sv1)) @ m
    m = np.kron(G.RY(3).matrix(), G.RX(1).matrix()) @ m
    m = G.ZZ(5).matrix() @ m
    m = (np.kron(G.I.matrix(), sv0) + np.kron(G.Z.matrix(), sv1)) @ m
    v = m[:, 0]
    assert np.allclose(v, v1)

    circ2 = circ.hermitian()
    s1.reset()
    s1.apply_circuit(circ2, pr)
    m = np.conj(m.T)
    v1 = s1.get_qs()
    v = m[:, 0]
    assert np.allclose(v, v1)


def _test_set_and_get(virtual_qc):
    """
    test
    Description:
    Expectation:
    """
    sim = Simulator(virtual_qc, 1)
    qs1 = sim.get_qs()
    assert np.allclose(qs1, np.array([1, 0]))
    sim.set_qs(np.array([1, 1]))
    qs2 = sim.get_qs()
    assert np.allclose(qs2, np.array([1, 1]) / np.sqrt(2))


def _test_non_hermitian_grad_ops(virtual_qc):
    """
    test
    Description:
    Expectation:
    """
    sim = Simulator(virtual_qc, 1)
    c_r = Circuit().ry('b', 0)
    c_l = Circuit().rz('a', 0)
    grad_ops = sim.get_expectation_with_grad(ops.Hamiltonian(ops.QubitOperator('')), c_r, c_l)
    f, g = grad_ops(np.array([1.2, 2.3]))
    f = f[0, 0]
    g = g[0, 0]
    f_exp = np.exp(1j * 2.3 / 2) * np.cos(1.2 / 2)
    g1 = -0.5 * np.exp(1j * 2.3 / 2) * np.sin(1.2 / 2)
    g2 = 1j / 2 * np.exp(1j * 2.3 / 2) * np.cos(1.2 / 2)
    assert np.allclose(f, f_exp)
    assert np.allclose(g, np.array([g1, g2]))


def generate_test_circuit():
    tmpg = G.RX('a')

    def rx_matrix_generator(x):
        return tmpg.matrix({'a': x})

    def rx_diff_matrix_generator(x):
        return tmpg.diff_matrix({'a': x}, 'a')

    c = Circuit()
    c += UN(G.H, 3)
    c.x(0).y(1).z(2)
    c += G.SWAP([0, 2], 1)
    c += UN(G.X, 3)
    c += G.ISWAP([0, 1], 2)
    c.rx(1.2, 0).ry(2.3, 1).rz(3.4, 2)
    c.x(0, 1).x(1, 2).x(0, 2)
    c += G.PhaseShift(1.3).on(0, [1, 2])
    c += UN(G.H, 3)
    c += UN(G.S, 3)
    c += qft(range(3))
    c += G.gene_univ_parameterized_gate('fake_x', rx_matrix_generator, rx_diff_matrix_generator)('a').on(0)
    c += G.RX('b').on(1, 2)
    c += G.RX('c').on(2, 0)
    c += UN(G.H, 3)
    c += UN(G.T, 3)
    c += G.UnivMathGate('fake_XX', G.XX(1.2).matrix()).on([0, 1])
    c += G.YY(2.3).on([1, 2])
    c += G.ZZ(3.4).on([0, 2])
    c += G.UnivMathGate('myX', G.X.matrix()).on(0)
    c += G.Power(G.X, 1.2).on(1)
    return c


def _test_all_gate_with_simulator(virtual_qc):
    """
    test
    Description:
    Expectation:
    """
    c = generate_test_circuit()
    qs = c.get_qs(backend=virtual_qc, pr={'a': 1, 'b': 2, 'c': 3})
    qs_exp = np.array(
        [
            0.09742526 + 0.00536111j,
            -0.17279339 - 0.32080812j,
            0.03473879 - 0.22046017j,
            -0.0990812 + 0.05735119j,
            -0.11858329 - 0.05715877j,
            0.37406968 + 0.19326249j,
            0.46926914 + 0.52135788j,
            -0.17429908 + 0.27887826j,
        ]
    )
    assert np.allclose(qs, qs_exp)
    sim = Simulator(virtual_qc, c.n_qubits)
    ham = ops.Hamiltonian(ops.QubitOperator('Z0'))
    grad_ops = sim.get_expectation_with_grad(ham, c)
    p0 = np.array([1, 2, 3])
    f1, g1 = grad_ops(p0)
    delta = 0.0001
    p1 = np.array([1 + delta, 2, 3])
    f2, g2 = grad_ops(p1)
    g_a = ((f2 - f1) / delta)[0, 0]
    g_a_1 = g1[0, 0, 0]
    g_a_2 = g2[0, 0, 0]
    assert np.allclose(g_a, g_a_1, atol=1e-4)
    assert np.allclose(g_a, g_a_2, atol=1e-4)
    assert np.allclose(g_a_1, g_a_2, atol=1e-4)


@pytest.mark.skipif(not _has_mindspore, reason='MindSpore is not installed')
def _test_optimization_with_custom_gate(virtual_qc):
    """
    test
    Description:
    Expectation:
    """
    if not _has_mindspore:  # NB: take care to avoid errors with 'ms' module below
        return

    def _matrix(theta):
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

    def _diff_matrix(theta):
        return 0.5 * np.array(
            [[-np.sin(theta / 2), -1j * np.cos(theta / 2)], [-1j * np.cos(theta / 2), -np.sin(theta / 2)]]
        )

    h = G.UnivMathGate('H', G.H.matrix())
    rx = G.gene_univ_parameterized_gate('RX', _matrix, _diff_matrix)

    c1 = Circuit() + G.RY(3.4).on(0) + h.on(0) + rx('a').on(0)
    c2 = Circuit() + G.RY(3.4).on(0) + G.H.on(0) + G.RX('a').on(0)

    sim = Simulator(virtual_qc, 1)
    ham = Hamiltonian(QubitOperator('Z0'))
    grad_ops1 = sim.get_expectation_with_grad(ham, c1)
    grad_ops2 = sim.get_expectation_with_grad(ham, c2)
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
    init_data = ms.Tensor(np.array([1.2]).astype(np.float32))
    net1 = MQAnsatzOnlyLayer(grad_ops1, weight=init_data)
    net2 = MQAnsatzOnlyLayer(grad_ops2, weight=init_data)
    opti1 = ms.nn.Adam(net1.trainable_params(), learning_rate=0.1)
    opti2 = ms.nn.Adam(net2.trainable_params(), learning_rate=0.1)
    train1 = ms.nn.TrainOneStepCell(net1, opti1)
    train2 = ms.nn.TrainOneStepCell(net2, opti2)
    for i in range(10):
        train1()
        train2()
    assert np.allclose(train1().asnumpy(), train2().asnumpy())


def _test_fid():
    """
    test
    Description:
    Expectation:
    """
    sim1 = Simulator('projectq', 1)
    prep_circ = Circuit().h(0)
    ansatz = Circuit().ry('a', 0).rz('b', 0).ry('c', 0)
    sim1.apply_circuit(prep_circ)
    sim2 = Simulator('projectq', 1)
    ham = Hamiltonian(QubitOperator(""))
    grad_ops = sim2.get_expectation_with_grad(ham, ansatz, Circuit(), simulator_left=sim1)
    f, _ = grad_ops(np.array([7.902762e-01, 2.139225e-04, 7.795934e-01]))
    assert np.allclose(np.abs(f), np.array([1]))


def test_virtual_quantum_computer():
    """
    test virtual quantum computer

    Description: Test mindquantum supported virtual quantum computers
    Expectation:
    """
    vqcs = ['projectq']
    for virtual_qc in vqcs:
        _test_init_reset(virtual_qc)
        _test_apply_circuit_and_hermitian(virtual_qc)
        _test_set_and_get(virtual_qc)
        _test_non_hermitian_grad_ops(virtual_qc)
        _test_all_gate_with_simulator(virtual_qc)
        _test_optimization_with_custom_gate(virtual_qc)
        _test_fid()


def test_non_hermitian_grad_ops():
    """
    Description: test non hermitian grad ops
    Expectation: success.
    """
    c1 = Circuit([G.RX('a').on(0)])
    c2 = Circuit([G.RY('b').on(0)])
    ham = Hamiltonian(csr_matrix([[1, 2], [3, 4]]))
    sim = Simulator('projectq', 1)
    grad_ops = sim.get_expectation_with_grad(ham, c2, c1)
    f, _ = grad_ops(np.array([1, 2]))
    f_exp = np.conj(G.RX(2).matrix().T) @ ham.sparse_mat.toarray() @ G.RY(1).matrix()
    f_exp = f_exp[0, 0]
    assert np.allclose(f, f_exp)


def test_inner_product():
    """
    Description: test inner product of two simulator
    Expectation: success.
    """
    sim1 = Simulator('projectq', 1)
    sim1.apply_gate(G.RX(1.2).on(0))
    sim2 = Simulator('projectq', 1)
    sim2.apply_gate(G.RY(2.1).on(0))
    val_exp = np.vdot(sim1.get_qs(), sim2.get_qs())
    val = inner_product(sim1, sim2)
    assert np.allclose(val_exp, val)


def test_copy():
    """
    Description: test copy a simulator
    Expectation: success.
    """
    sim = Simulator('projectq', 1)
    sim.apply_gate(G.RX(1).on(0))
    sim.flush()
    sim2 = sim.copy()
    sim2.apply_gate(G.RX(-1).on(0))
    sim.reset()
    qs1 = sim.get_qs()
    qs2 = sim2.get_qs()
    assert np.allclose(qs1, qs2)
