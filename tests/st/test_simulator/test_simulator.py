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
"""Test simulator."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mindquantum.core.operators as ops
from mindquantum.algorithm.library import qft
from mindquantum.config import Context
from mindquantum.core import gates as G
from mindquantum.core.circuit import UN, Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.simulator import Simulator, get_supported_simulator, inner_product
from mindquantum.utils import random_circuit

_HAS_MINDSPORE = True
try:
    import mindspore as ms

    from mindquantum.framework.layer import (  # pylint: disable=ungrouped-imports
        MQAnsatzOnlyLayer,
    )

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
except ImportError:
    _HAS_MINDSPORE = False

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
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_init_reset(virtual_qc, dtype):
    """
    test
    Description:
    Expectation:
    """
    Context.set_dtype(dtype)
    s1 = Simulator(virtual_qc, 2)
    circ = Circuit().h(0).h(1)
    v1 = s1.get_qs()
    s1.apply_circuit(circ)
    s1.reset()
    v3 = s1.get_qs()
    v = np.array([1, 0, 0, 0], dtype=np.complex128)
    if virtual_qc == "mqmatrix":
        assert np.allclose(v1, np.outer(v, v.conj()))
        assert np.allclose(v1, v3)
    else:
        assert np.allclose(v1, v)
        assert np.allclose(v1, v3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_apply_circuit_and_hermitian(virtual_qc, dtype):
    """
    test
    Description:
    Expectation:
    """
    Context.set_dtype(dtype)
    sv0 = np.array([[1, 0], [0, 0]])
    sv1 = np.array([[0, 0], [0, 1]])
    circ = Circuit()
    circ.ry(1.2, 0).ry(3.4, 1)
    circ.h(0).h(1)
    circ.x(1, 0)
    circ.rx('a', 0).ry('b', 1)
    circ.rzz({'c': 2}, (0, 1)).z(1, 0)
    s1 = Simulator(virtual_qc, circ.n_qubits)
    pr = PR({'a': 1, 'b': 3, 'c': 5})
    s1.apply_circuit(circ, pr)
    v1 = s1.get_qs()
    matrix = np.kron(G.RY(3.4).matrix(), G.RY(1.2).matrix())
    matrix = np.kron(G.H.matrix(), G.H.matrix()) @ matrix
    matrix = (np.kron(G.I.matrix(), sv0) + np.kron(G.X.matrix(), sv1)) @ matrix
    matrix = np.kron(G.RY(3).matrix(), G.RX(1).matrix()) @ matrix
    matrix = G.Rzz(10).matrix() @ matrix
    matrix = (np.kron(G.I.matrix(), sv0) + np.kron(G.Z.matrix(), sv1)) @ matrix
    v = matrix[:, 0]
    if virtual_qc == "mqmatrix":
        m = np.outer(v, v.conj())
        assert np.allclose(m, v1)
    else:
        assert np.allclose(v, v1)

    circ2 = circ.hermitian()
    s1.reset()
    s1.apply_circuit(circ2, pr)
    matrix = np.conj(matrix.T)
    v1 = s1.get_qs()
    v = matrix[:, 0]
    if virtual_qc == "mqmatrix":
        m = np.outer(v, v.conj())
        assert np.allclose(m, v1)
    else:
        assert np.allclose(v, v1)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_set_and_get(virtual_qc, dtype):
    """
    test
    Description:
    Expectation:
    """
    Context.set_dtype(dtype)
    sim = Simulator(virtual_qc, 1)
    qs1 = sim.get_qs()
    if virtual_qc == "mqmatrix":
        assert np.allclose(qs1, np.array([[1, 0], [0, 0]]))
    else:
        assert np.allclose(qs1, np.array([1, 0]))
    sim.set_qs(np.array([1, 1]))
    qs2 = sim.get_qs()
    if virtual_qc == "mqmatrix":
        assert np.allclose(qs2, np.array([[0.5, 0.5], [0.5, 0.5]]))
    else:
        assert np.allclose(qs2, np.array([1, 1]) / np.sqrt(2))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", [i for i in get_supported_simulator() if i != 'mqmatrix'])
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_non_hermitian_grad_ops1(virtual_qc, dtype):
    """
    test
    Description:
    Expectation:
    """
    Context.set_dtype(dtype)
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
    """
    Description:
    Expectation:
    """

    def rx_matrix_generator(x):
        return np.array([[np.cos(x / 2), -1j * np.sin(x / 2)], [-1j * np.sin(x / 2), np.cos(x / 2)]])

    def rx_diff_matrix_generator(x):
        return np.array([[np.sin(x / 2), 1j * np.cos(x / 2)], [1j * np.cos(x / 2), np.sin(x / 2)]]) / -2

    circuit = Circuit()
    circuit += UN(G.H, 3)
    circuit.x(0).y(1).z(2)
    circuit += G.SWAP([0, 2], 1)
    circuit += UN(G.X, 3)
    circuit += G.ISWAP([0, 1], 2)
    circuit.rx(1.2, 0).ry(2.3, 1).rz(3.4, 2)
    circuit.x(0, 1).x(1, 2).x(0, 2)
    circuit += G.PhaseShift(1.3).on(0, [1, 2])
    circuit += UN(G.H, 3)
    circuit += UN(G.S, 3)
    circuit += qft(range(3))
    circuit += G.gene_univ_parameterized_gate('fake_x', rx_matrix_generator, rx_diff_matrix_generator)('a').on(0)
    circuit += G.RX('b').on(1, 2)
    circuit += G.RX('c').on(2, 0)
    circuit += UN(G.H, 3)
    circuit += UN(G.T, 3)
    circuit += G.UnivMathGate('fake_Rxx', G.Rxx(2.4).matrix()).on([0, 1])
    circuit += G.Ryy(4.6).on([1, 2])
    circuit += G.Rzz(6.8).on([0, 2])
    circuit += G.UnivMathGate('myX', G.X.matrix()).on(0)
    circuit += G.Power(G.X, 1.2).on(1)
    return circuit


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_all_gate_with_simulator(virtual_qc, dtype):  # pylint: disable=too-many-locals
    """
    Description:
    Expectation:
    """
    Context.set_dtype(dtype)
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
    if virtual_qc == "mqmatrix":
        assert np.allclose(qs, np.outer(qs_exp, qs_exp.conj()))
    else:
        assert np.allclose(qs, qs_exp)
    sim = Simulator(virtual_qc, c.n_qubits)
    ham = ops.Hamiltonian(ops.QubitOperator('Z0'))
    grad_ops = sim.get_expectation_with_grad(ham, c)
    p0 = np.array([1, 2, 3])
    f1, g1 = grad_ops(p0)
    delta = 0.00001
    p1 = np.array([1 + delta, 2, 3])
    f2, g2 = grad_ops(p1)
    g_a = ((f2 - f1) / delta)[0, 0]
    g_a_1 = g1[0, 0, 0]
    g_a_2 = g2[0, 0, 0]
    atol = 1e-3
    if dtype == 'float':
        atol = 1e-1
    assert np.allclose(g_a, g_a_1, atol=atol)
    assert np.allclose(g_a, g_a_2, atol=atol)
    assert np.allclose(g_a_1, g_a_2, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.parametrize("dtype", ['float', 'double'])
@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_optimization_with_custom_gate(virtual_qc, dtype):  # pylint: disable=too-many-locals
    """
    Description:
    Expectation:
    """
    Context.set_dtype(dtype)
    if not _HAS_MINDSPORE:  # NB: take care to avoid errors with 'ms' module below
        return

    def _matrix(theta):
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

    def _diff_matrix(theta):
        return 0.5 * np.array(
            [[-np.sin(theta / 2), -1j * np.cos(theta / 2)], [-1j * np.cos(theta / 2), -np.sin(theta / 2)]]
        )

    h = G.UnivMathGate('H', G.H.matrix())
    rx = G.gene_univ_parameterized_gate('RX', _matrix, _diff_matrix)

    circuit1 = Circuit() + G.RY(3.4).on(0) + h.on(0) + rx('a').on(0)
    circuit2 = Circuit() + G.RY(3.4).on(0) + G.H.on(0) + G.RX('a').on(0)

    sim = Simulator(virtual_qc, 1)
    ham = Hamiltonian(QubitOperator('Z0'))
    grad_ops1 = sim.get_expectation_with_grad(ham, circuit1)
    grad_ops2 = sim.get_expectation_with_grad(ham, circuit2)
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
    init_data = ms.Tensor(np.array([1.2]).astype(np.float32))
    net1 = MQAnsatzOnlyLayer(grad_ops1, weight=init_data)
    net2 = MQAnsatzOnlyLayer(grad_ops2, weight=init_data)
    opti1 = ms.nn.Adam(net1.trainable_params(), learning_rate=0.1)
    opti2 = ms.nn.Adam(net2.trainable_params(), learning_rate=0.1)
    train1 = ms.nn.TrainOneStepCell(net1, opti1)
    train2 = ms.nn.TrainOneStepCell(net2, opti2)
    for _ in range(10):
        train1()
        train2()
    assert np.allclose(train1().asnumpy(), train2().asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", [i for i in get_supported_simulator() if i != 'mqmatrix'])
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_fid(virtual_qc, dtype):
    """
    Description:
    Expectation:
    """
    Context.set_dtype(dtype)
    sim1 = Simulator(virtual_qc, 1)
    prep_circ = Circuit().h(0)
    ansatz = Circuit().ry('a', 0).rz('b', 0).ry('c', 0)
    sim1.apply_circuit(prep_circ)
    sim2 = Simulator(virtual_qc, 1)
    ham = Hamiltonian(QubitOperator(""))
    grad_ops = sim2.get_expectation_with_grad(ham, ansatz, Circuit(), simulator_left=sim1)
    f, _ = grad_ops(np.array([7.902762e-01, 2.139225e-04, 7.795934e-01]))
    assert np.allclose(np.abs(f), np.array([1]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", [i for i in get_supported_simulator() if i != 'mqmatrix'])
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_non_hermitian_grad_ops2(virtual_qc, dtype):
    """
    Description: test non hermitian grad ops
    Expectation: success.
    """
    Context.set_dtype(dtype)
    circuit1 = Circuit([G.RX('a').on(0)])
    circuit2 = Circuit([G.RY('b').on(0)])
    ham = Hamiltonian(csr_matrix([[1, 2], [3, 4]]))
    sim = Simulator(virtual_qc, 1)
    grad_ops = sim.get_expectation_with_grad(ham, circuit2, circuit1)
    f, _ = grad_ops(np.array([1, 2]))
    f_exp = np.conj(G.RX(2).matrix().T) @ ham.sparse_mat.toarray() @ G.RY(1).matrix()
    f_exp = f_exp[0, 0]
    assert np.allclose(f, f_exp)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", [i for i in get_supported_simulator() if i != 'mqmatrix'])
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_inner_product(virtual_qc, dtype):
    """
    Description: test inner product of two simulator
    Expectation: success.
    """
    Context.set_dtype(dtype)
    sim1 = Simulator(virtual_qc, 1)
    sim1.apply_gate(G.RX(1.2).on(0))
    sim2 = Simulator(virtual_qc, 1)
    sim2.apply_gate(G.RY(2.1).on(0))
    val_exp = np.vdot(sim1.get_qs(), sim2.get_qs())
    val = inner_product(sim1, sim2)
    assert np.allclose(val_exp, val)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_copy(virtual_qc, dtype):
    """
    Description: test copy a simulator
    Expectation: success.
    """
    Context.set_dtype(dtype)
    sim = Simulator(virtual_qc, 1)
    sim.apply_gate(G.RX(1).on(0))
    sim2 = sim.copy()
    sim2.apply_gate(G.RX(-1).on(0))
    sim.reset()
    qs1 = sim.get_qs()
    qs2 = sim2.get_qs()
    if virtual_qc == 'mqmatrix' and dtype == 'float':
        assert np.allclose(qs1, qs2, atol=1.0e-6)
    else:
        assert np.allclose(qs1, qs2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_univ_order(virtual_qc, dtype):
    """
    Description: test order of univ math gate.
    Expectation: success.
    """
    Context.set_dtype(dtype)
    r_c = random_circuit(2, 100)
    if virtual_qc == 'mqmatrix':
        u = r_c.matrix(backend='mqvector')
        assert np.allclose(r_c.get_qs(backend=virtual_qc), np.outer(u[:, 0], np.conj(u[:, 0])))
    else:
        u = r_c.matrix(backend=virtual_qc)
        assert np.allclose(r_c.get_qs(backend=virtual_qc), u[:, 0])
    g = G.UnivMathGate('u', u)
    c0 = Circuit([g.on([0, 1])])
    c1 = Circuit([g.on([1, 0])])
    if virtual_qc == 'mqmatrix':
        assert np.allclose(c0.get_qs(backend=virtual_qc), np.outer(u[:, 0], np.conj(u[:, 0])))
        v_tmp = np.array([u[0, 0], u[2, 0], u[1, 0], u[3, 0]])
        assert np.allclose(c1.get_qs(backend=virtual_qc), np.outer(v_tmp, np.conj(v_tmp)))
    else:
        assert np.allclose(c0.get_qs(backend=virtual_qc), u[:, 0])
        assert np.allclose(c1.get_qs(backend=virtual_qc), np.array([u[0, 0], u[2, 0], u[1, 0], u[3, 0]]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_multi_params_gate(virtual_qc, dtype):
    """
    Description: test multi params gate
    Expectation: success.
    """
    Context.set_dtype(dtype)
    sim = Simulator(virtual_qc, 2)
    circ = Circuit() + G.U3('a', 'b', 1.0).on(0) + G.U3('c', 'd', 2.0).on(1) + G.X.on(0, 1)
    circ += G.FSim('e', 3.0).on([0, 1])
    p0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sim.apply_circuit(circ, pr=dict(zip(circ.params_name, p0)))
    qs_exp = np.array([0.06207773 + 0.0j, 0.12413139 + 0.44906334j, 0.10068061 - 0.05143708j, 0.65995413 + 0.57511569j])
    if virtual_qc == "mqmatrix":
        assert np.allclose(np.outer(qs_exp, qs_exp.conj()), sim.get_qs())
    else:
        assert np.allclose(qs_exp, sim.get_qs())
    sim.reset()
    ham = Hamiltonian(QubitOperator("X0 Y1"))
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    f, g = grad_ops(p0)
    f_exp = np.array([[-0.0317901 + 0.0j]])
    g_exp = np.array(
        [
            [
                [
                    -2.27903141e-01 + 0.0j,
                    4.32795462e-18 + 0.0j,
                    -6.63057399e-01 + 0.0j,
                    9.97267089e-02 + 0.0j,
                    -4.08568259e-01 + 0.0j,
                ]
            ]
        ]
    )
    assert np.allclose(f, f_exp, atol=1e-6)
    assert np.allclose(g, g_exp, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", get_supported_simulator())
@pytest.mark.parametrize("dtype", ['float', 'double'])
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_custom_gate_in_parallel(virtual_qc, dtype):
    """
    Features: parallel custom gate.
    Description: test custom gate in parallel mode.
    Expectation: success.
    """
    Context.set_dtype(dtype)
    circ = generate_test_circuit().as_encoder()
    sim = Simulator(virtual_qc, circ.n_qubits)
    ham = [Hamiltonian(QubitOperator('Y0')), Hamiltonian(QubitOperator('X2'))]
    np.random.seed(42)
    p0 = np.random.uniform(0, 1, size=(2, len(circ.params_name)))
    grad_ops = sim.get_expectation_with_grad(ham, circ, parallel_worker=4)
    f, g = grad_ops(p0)
    f_sum_exp = 0.8396650072427185
    g_sum_exp = 0.06041889360878677
    assert np.allclose(np.sum(f), f_sum_exp)
    assert np.allclose(np.sum(g), g_sum_exp)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("virtual_qc", [i for i in get_supported_simulator() if i != 'mqmatrix'])
@pytest.mark.parametrize("dtype", ['float', 'double'])
def test_cd_term(virtual_qc, dtype):
    """
    Description:
    Expectation:
    """
    Context.set_dtype(dtype)
    cd_term = [G.Rxy, G.Rxz, G.Ryz]
    for g in cd_term:
        cd_gate = g(1.0).on([0, 1])
        circ = Circuit() + cd_gate
        decomps_circ = cd_gate.__decompose__()[0]
        m1 = circ.matrix(backend=virtual_qc)
        m2 = decomps_circ.matrix(backend=virtual_qc)
        m = np.abs(m1 - m2)
        assert np.allclose(m, np.zeros_like(m), atol=1e-6)
