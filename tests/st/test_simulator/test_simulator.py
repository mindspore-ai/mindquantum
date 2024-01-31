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
import platform
import subprocess

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mindquantum as mq
import mindquantum.core.operators as ops
from mindquantum.algorithm.library import qft
from mindquantum.algorithm.nisq import Ansatz6
from mindquantum.core import gates as G
from mindquantum.core.circuit import (
    UN,
    BitFlipAdder,
    Circuit,
    MeasureAccepter,
    MixerAdder,
)
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.simulator import NoiseBackend, Simulator, inner_product
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR
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

_HAS_GPU = False

try:
    subprocess.check_output('nvidia-smi')
    _HAS_GPU = True
except FileNotFoundError:
    _HAS_GPU = False


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(platform.system() != 'Linux', reason='GPU backend only available for linux.')
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
@pytest.mark.parametrize("dtype", [mq.complex128, mq.complex64])
def test_gpu(dtype):
    """
    test gpu
    Description: to make sure gpu platform was tested.
    Expectation: gpu backend not available.
    """
    sim = Simulator('mqvector_gpu', 1, dtype=dtype)
    sim.apply_circuit(Circuit().h(0))
    v = sim.get_qs()
    assert np.allclose(v, np.ones_like(v) / np.sqrt(2))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_init_reset(config):
    """
    test
    Description:
    Expectation:
    """
    virtual_qc, dtype = config
    s1 = Simulator(virtual_qc, 2, dtype=dtype)
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
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_apply_circuit_and_hermitian(config):
    """
    test
    Description:
    Expectation:
    """
    virtual_qc, dtype = config
    sv0 = np.array([[1, 0], [0, 0]])
    sv1 = np.array([[0, 0], [0, 1]])
    circ = Circuit()
    circ.ry(1.2, 0).ry(3.4, 1)
    circ.h(0).h(1)
    circ.x(1, 0)
    circ.rx('a', 0).ry('b', 1)
    circ.rzz({'c': 2}, (0, 1)).z(1, 0)
    s1 = Simulator(virtual_qc, circ.n_qubits, dtype=dtype)
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
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_set_and_get(config):
    """
    test
    Description:
    Expectation:
    """
    virtual_qc, dtype = config
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    qs1 = sim.get_qs()
    if virtual_qc == "mqmatrix":
        assert np.allclose(qs1, np.array([[1, 0], [0, 0]]))
    else:
        assert np.allclose(qs1, np.array([1, 0]))
    sim.set_qs(np.array([1, 1]))
    qs2 = sim.get_qs()
    if virtual_qc == "mqmatrix":
        assert np.allclose(qs2, np.array([[0.5, 0.5], [0.5, 0.5]]))
        sim.set_qs(np.array([[1, 1], [1, 1]]))
        qs2 = sim.get_qs()
        assert np.allclose(qs2, np.array([[0.5, 0.5], [0.5, 0.5]]))
    else:
        assert np.allclose(qs2, np.array([1, 1]) / np.sqrt(2))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_non_hermitian_grad_ops1(config):
    """
    test
    Description:
    Expectation:
    """
    virtual_qc, dtype = config
    if virtual_qc == 'mqmatrix':
        return
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    c_r = Circuit().ry('b', 0)
    c_l = Circuit().rz('a', 0)
    grad_ops = sim.get_expectation_with_grad(ops.Hamiltonian(ops.QubitOperator('').astype(dtype)), c_r, c_l)
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
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_all_gate_with_simulator(config):  # pylint: disable=too-many-locals
    """
    Description:
    Expectation:
    """
    virtual_qc, dtype = config
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
    sim = Simulator(virtual_qc, c.n_qubits, dtype=dtype)
    ham = ops.Hamiltonian(ops.QubitOperator('Z0'), dtype=dtype)
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
    if dtype == mq.complex64:
        atol = 1e-1
    assert np.allclose(g_a, g_a_1, atol=atol)
    assert np.allclose(g_a, g_a_2, atol=atol)
    assert np.allclose(g_a_1, g_a_2, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_optimization_with_custom_gate(config):  # pylint: disable=too-many-locals
    """
    Description:
    Expectation:
    """
    virtual_qc, dtype = config
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

    sim = Simulator(virtual_qc, 1, dtype=dtype)
    ham = Hamiltonian(QubitOperator('Z0'), dtype=dtype)
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
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_fid(config):
    """
    Description:
    Expectation:
    """
    virtual_qc, dtype = config
    if virtual_qc == 'mqmatrix':
        return
    sim1 = Simulator(virtual_qc, 1, dtype=dtype)
    prep_circ = Circuit().h(0)
    ansatz = Circuit().ry('a', 0).rz('b', 0).ry('c', 0)
    sim1.apply_circuit(prep_circ)
    sim2 = Simulator(virtual_qc, 1, dtype=dtype)
    ham = Hamiltonian(QubitOperator("").astype(dtype))
    grad_ops = sim2.get_expectation_with_grad(ham, ansatz, Circuit(), simulator_left=sim1)
    f, _ = grad_ops(np.array([7.902762e-01, 2.139225e-04, 7.795934e-01]))
    assert np.allclose(np.abs(f), np.array([1]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_non_hermitian_grad_ops2(config):
    """
    Description: test non hermitian grad ops
    Expectation: success.
    """
    virtual_qc, dtype = config
    if virtual_qc == 'mqmatrix':
        return
    circuit1 = Circuit([G.RX('a').on(0)])
    circuit2 = Circuit([G.RY('b').on(0)])
    ham = Hamiltonian(csr_matrix([[1.0, 2.0], [3.0, 4.0]]), dtype=dtype)
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    grad_ops = sim.get_expectation_with_grad(ham, circuit2, circuit1)
    f, _ = grad_ops(np.array([1, 2]))
    f_exp = np.conj(G.RX(2).matrix().T) @ ham.sparse_mat.toarray() @ G.RY(1).matrix()
    f_exp = f_exp[0, 0]
    assert np.allclose(f, f_exp)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_csr_ham(config):
    """
    Description: test csr matrix hamiltonian
    Expectation: success.
    """
    virtual_qc, dtype = config
    circ = Circuit([G.RX('a').on(0)])
    circ += G.RY('b').on(0)
    ham = Hamiltonian(csr_matrix([[1.0, 2.0], [2.0, 4.0]]), dtype=dtype)
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    f, g = grad_ops(np.array([1, 2]))
    f_exp = np.conj(circ.matrix(np.array([1, 2])).T) @ ham.sparse_mat.toarray() @ circ.matrix(np.array([1, 2]))
    f_exp = f_exp[0, 0]
    assert np.allclose(f, f_exp)
    if virtual_qc == 'mqmatrix':
        sim2 = Simulator('mqvector', 1, dtype=dtype)
        grad_ops2 = sim2.get_expectation_with_grad(ham, circ)
        _, g2 = grad_ops2(np.array([1, 2]))
        assert np.allclose(g, g2)
    sim.apply_circuit(circ, np.array([1, 2]))
    sim.apply_hamiltonian(ham)
    qs = sim.get_qs()
    qs_exp = ham.sparse_mat.toarray() @ circ.matrix(np.array([1, 2])) @ np.array([1, 0])
    if virtual_qc == 'mqmatrix':
        assert np.allclose(qs, np.outer(qs_exp, qs_exp.conj()))
    else:
        assert np.allclose(qs, qs_exp)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_inner_product(config):
    """
    Description: test inner product of two simulator
    Expectation: success.
    """
    virtual_qc, dtype = config
    if virtual_qc == 'mqmatrix':
        return
    sim1 = Simulator(virtual_qc, 1, dtype=dtype)
    sim1.apply_gate(G.RX(1.2).on(0))
    sim2 = Simulator(virtual_qc, 1, dtype=dtype)
    sim2.apply_gate(G.RY(2.1).on(0))
    val_exp = np.vdot(sim1.get_qs(), sim2.get_qs())
    val = inner_product(sim1, sim2)
    assert np.allclose(val_exp, val)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_copy(config):
    """
    Description: test copy a simulator
    Expectation: success.
    """
    virtual_qc, dtype = config
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    sim.apply_gate(G.RX(1).on(0))
    sim2 = sim.copy()
    sim2.apply_gate(G.RX(-1).on(0))
    sim.reset()
    qs1 = sim.get_qs()
    qs2 = sim2.get_qs()
    if virtual_qc == 'mqmatrix' and dtype == mq.complex64:
        assert np.allclose(qs1, qs2, atol=1.0e-6)
    else:
        assert np.allclose(qs1, qs2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_univ_order(config):
    """
    Description: test order of univ math gate.
    Expectation: success.
    """
    virtual_qc, dtype = config
    r_c = random_circuit(2, 100)
    if virtual_qc == 'mqmatrix':
        u = r_c.matrix(backend='mqvector', dtype=dtype)
        assert np.allclose(r_c.get_qs(backend=virtual_qc, dtype=dtype), np.outer(u[:, 0], np.conj(u[:, 0])), atol=1e-6)
    else:
        u = r_c.matrix(backend=virtual_qc, dtype=dtype)
        assert np.allclose(r_c.get_qs(backend=virtual_qc, dtype=dtype), u[:, 0], atol=1e-6)
    g = G.UnivMathGate('u', u)
    c0 = Circuit([g.on([0, 1])])
    c1 = Circuit([g.on([1, 0])])
    if virtual_qc == 'mqmatrix':
        assert np.allclose(c0.get_qs(backend=virtual_qc, dtype=dtype), np.outer(u[:, 0], np.conj(u[:, 0])), atol=1e-6)
        v_tmp = np.array([u[0, 0], u[2, 0], u[1, 0], u[3, 0]])
        assert np.allclose(c1.get_qs(backend=virtual_qc, dtype=dtype), np.outer(v_tmp, np.conj(v_tmp)), atol=1e-6)
    else:
        assert np.allclose(c0.get_qs(backend=virtual_qc, dtype=dtype), u[:, 0], atol=1e-6)
        assert np.allclose(
            c1.get_qs(backend=virtual_qc, dtype=dtype), np.array([u[0, 0], u[2, 0], u[1, 0], u[3, 0]]), atol=1e-6
        )


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_multi_params_gate(config):
    """
    Description: test multi params gate
    Expectation: success.
    """
    virtual_qc, dtype = config
    sim = Simulator(virtual_qc, 2, dtype=dtype)
    circ = Circuit() + G.U3('a', 'b', 1.0).on(0) + G.U3('c', 'd', 2.0).on(1) + G.X.on(0, 1)
    circ += G.FSim('e', -3.0).on([0, 1])
    p0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sim.apply_circuit(circ, pr=dict(zip(circ.params_name, p0)))
    qs_exp = np.array([0.06207773 + 0.0j, 0.12413139 + 0.44906334j, 0.10068061 - 0.05143708j, 0.65995413 + 0.57511569j])
    if virtual_qc == "mqmatrix":
        assert np.allclose(np.outer(qs_exp, qs_exp.conj()), sim.get_qs())
    else:
        assert np.allclose(qs_exp, sim.get_qs())
    sim.reset()
    ham = Hamiltonian(QubitOperator("X0 Y1").astype(dtype))
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
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_custom_gate_in_parallel(config):
    """
    Features: parallel custom gate.
    Description: test custom gate in parallel mode.
    Expectation: success.
    """
    virtual_qc, dtype = config
    circ = generate_test_circuit().as_encoder()
    sim = Simulator(virtual_qc, circ.n_qubits, dtype=dtype)
    ham = [Hamiltonian(QubitOperator('Y0').astype(dtype)), Hamiltonian(QubitOperator('X2').astype(dtype))]
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
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_cd_term(config):
    """
    Description:
    Expectation:
    """
    virtual_qc, dtype = config
    if virtual_qc == 'mqmatrix':
        return
    cd_term = [G.Rxy, G.Rxz, G.Ryz]
    for g in cd_term:
        cd_gate = g(1.0).on([0, 1])
        circ = Circuit() + cd_gate
        decomps_circ = cd_gate.__decompose__()[0]
        m1 = circ.matrix(backend=virtual_qc, dtype=dtype)
        m2 = decomps_circ.matrix(backend=virtual_qc, dtype=dtype)
        m = np.abs(m1 - m2)
        assert np.allclose(m, np.zeros_like(m), atol=1e-6)


def custom_matrix(x):
    """Define matrix."""
    return np.array(
        [
            [np.exp(1j * 2 * x), 0, 0, 0, 0, 0, 0, 0],
            [0, np.exp(1j * 4 * x), 0, 0, 0, 0, 0, 0],
            [0, 0, np.exp(1j * 6 * x), 0, 0, 0, 0, 0],
            [0, 0, 0, np.exp(1j * 8 * x), 0, 0, 0, 0],
            [0, 0, 0, 0, np.exp(1j * 10 * x), 0, 0, 0],
            [0, 0, 0, 0, 0, np.exp(1j * 12 * x), 0, 0],
            [0, 0, 0, 0, 0, 0, np.exp(1j * 14 * x), 0],
            [0, 0, 0, 0, 0, 0, 0, np.exp(1j * 16 * x)],
        ],
        dtype=np.complex128,
    )


def custom_diff_matrix(x):
    """Define diff matrix."""
    return (
        np.array(
            [
                [2 * np.exp(1j * 2 * x), 0, 0, 0, 0, 0, 0, 0],
                [0, 4 * np.exp(1j * 4 * x), 0, 0, 0, 0, 0, 0],
                [0, 0, 6 * np.exp(1j * 6 * x), 0, 0, 0, 0, 0],
                [0, 0, 0, 8 * np.exp(1j * 8 * x), 0, 0, 0, 0],
                [0, 0, 0, 0, 10 * np.exp(1j * 10 * x), 0, 0, 0],
                [0, 0, 0, 0, 0, 12 * np.exp(1j * 12 * x), 0, 0],
                [0, 0, 0, 0, 0, 0, 14 * np.exp(1j * 14 * x), 0],
                [0, 0, 0, 0, 0, 0, 0, 16 * np.exp(1j * 16 * x)],
            ],
            dtype=np.complex128,
        )
        * 1j
    )


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_mul_qubit_gate(config):
    """
    Description: Test simulation on multiple qubit gate.
    Expectation: succeed.
    """
    virtual_qc, dtype = config
    rand_c = random_circuit(3, 20, seed=42)
    m = rand_c.matrix()
    g = G.UnivMathGate('m', m)
    q = G.gene_univ_parameterized_gate('q', custom_matrix, custom_diff_matrix)
    circ = UN(G.H, 3) + q('a').on([0, 1, 2])
    sim = Simulator(virtual_qc, 3, dtype=dtype)
    ham = Hamiltonian(QubitOperator("X0"), dtype=dtype)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    f1, g1 = grad_ops(np.array([2.3]))
    f2, _ = grad_ops(np.array([2.3001]))
    g = (f2 - f1) / 0.0001
    assert np.allclose(f1, -0.11215253 + 0.0j)
    assert np.allclose(g1, 1.98738201 + 0.0j)
    assert np.allclose(g1, g, atol=1e-2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "virtual_qc",
    [
        'mqvector',
        pytest.param('mqvector_gpu', marks=pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')),
    ],
)
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_non_hermitian_expectation(virtual_qc, dtype):
    """
    Description: Test get expectation with non hermitian situation.
    Expectation: succeed.
    """
    sim = Simulator(virtual_qc, 1, dtype=dtype)
    sim.apply_circuit(Circuit().ry(1.2, 0))
    ham = Hamiltonian(QubitOperator('Z0'), dtype=dtype)
    e1 = sim.get_expectation(ham, Circuit().rx('a', 0), Circuit().ry(2.3, 0), pr={'a': 2.4})
    sim1, sim2 = Simulator(virtual_qc, 1, dtype=dtype), Simulator(virtual_qc, 1, dtype=dtype)
    sim1.apply_circuit(Circuit().ry(1.2, 0).rx(2.4, 0))
    sim2.apply_circuit(Circuit().ry(1.2, 0).ry(2.3, 0))
    sim1.apply_hamiltonian(ham)
    e2 = inner_product(sim2, sim1)
    assert np.allclose(e1, e2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_noise_simulator(config):
    """
    Description: Test noise simulator.
    Expectation: succeed.
    """
    virtual_qc, dtype = config
    circ = Circuit().h(0).x(1, 0).measure_all()
    adder = MixerAdder(
        [
            MeasureAccepter(),
            BitFlipAdder(0.2),
        ],
        add_after=False,
    )
    sim = Simulator(NoiseBackend(virtual_qc, 2, adder=adder, dtype=dtype))
    res = sim.sampling(circ, seed=42, shots=5000)
    if virtual_qc.startswith('mqvector'):
        assert res.data['00'] == 1701
    elif virtual_qc.startswith('mqmatrix'):
        assert res.data['00'] == 1684


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_measurement_reset(config):
    """
    Description: Test measurement_reset.
    Expectation: succeed.
    """
    virtual_qc, _ = config
    c = Circuit().rx(2.2, 0).ry(1.2, 1)
    c.measure(0, reset_to=1)
    c.measure(1, reset_to=0)
    c.measure('q0_1', 0)
    c.measure('q1_1', 1)
    sim = Simulator(virtual_qc, c.n_qubits)
    res = sim.sampling(c, shots=100, seed=123)
    assert all(i[:2] == '01' for i in res.data.keys())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_parameter_shift_rule():
    """
    Description: Test fix of parameter shift rule bug.
    Expectation: succeed.
    """
    # pylint: disable=too-many-locals
    # TODO: controlled parameterized not supported for parameter shift currently.
    c = Ansatz6(3, 1, 'e').circuit.as_encoder() + Ansatz6(3, 1, 'a').circuit.as_ansatz()
    new_c = Circuit()
    for i in c:
        if isinstance(i, G.RX) and i.ctrl_qubits:
            new_c += G.X.on(i.obj_qubits, i.ctrl_qubits)
        else:
            new_c += i
    c = new_c
    sim1 = Simulator('mqvector', c.n_qubits)
    sim2 = Simulator('mqvector', c.n_qubits)
    ham = Hamiltonian(QubitOperator('Z0 Y1 X2'))
    grad_ops1 = sim1.get_expectation_with_grad(ham, c, parallel_worker=5)
    noise_c = c + G.AmplitudeDampingChannel(0.0).on(0)
    grad_ops2 = sim2.get_expectation_with_grad(ham, noise_c, parallel_worker=5)
    p_e = np.random.uniform(-3, 3, size=(5, len(c.encoder_params_name)))
    p_a = np.random.uniform(-3, 3, size=len(c.ansatz_params_name))
    for i in range(3):
        f1, ge_1, ga_1 = grad_ops1(p_e, p_a)
        f2, ge_2, ga_2 = grad_ops2(p_e, p_a)
        assert np.allclose(f1, f2)
        assert np.allclose(ge_1, ge_2)
        assert np.allclose(ga_1, ga_2)
