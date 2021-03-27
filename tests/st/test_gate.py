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
"""Test gate."""

from projectq.ops import QubitOperator
import qutip as qt
import numpy as np
from mindquantum.gate import Hamiltonian
import mindquantum.gate as G


def test_hamiltonian():
    """Test hamiltonian"""
    ham = Hamiltonian(QubitOperator('Z0 Y1', 0.3))
    assert ham.ham_termlist == [(((0, 'Z'), (1, 'Y')), 0.3)]
    assert ham.mindspore_data() == {
        'hams_pauli_coeff': [0.3],
        'hams_pauli_word': [['Z', 'Y']],
        'hams_pauli_qubit': [[0, 1]]
    }


def test_rotate_pauli():
    gates = {
        'rx': [G.RX('angle').on(0), qt.qip.operations.rx],
        'ry': [G.RY('angle').on(0), qt.qip.operations.ry],
        'rz': [G.RZ('angle').on(0), qt.qip.operations.rz]
    }
    angle = 0.5
    for name, rs in gates.items():
        assert np.allclose(rs[0].matrix({'angle': angle}),
                           rs[1](angle).data.toarray())
        assert np.allclose(rs[0].diff_matrix({'angle': angle}),
                           0.5 * rs[1](angle + np.pi).data.toarray())
        assert np.allclose(rs[0].hermitian().matrix({'angle': angle}),
                           rs[1](-angle).data.toarray())
        assert np.allclose(rs[0].hermitian().diff_matrix({'angle': angle}),
                           0.5 * rs[1](-angle - np.pi).data.toarray())


def test_phase_shift():
    angle = 0.5
    assert np.allclose(
        G.PhaseShift(angle).matrix(),
        qt.qip.operations.phasegate(angle).data.toarray())


def test_trap_ion_gate():
    angle = 0.5
    xx = [
        G.XX("angle").on(0), lambda angle:
        (-1j * angle * qt.tensor(qt.sigmax(), qt.sigmax())).expm()
    ]
    yy = [
        G.YY("angle").on(0), lambda angle:
        (-1j * angle * qt.tensor(qt.sigmay(), qt.sigmay())).expm()
    ]
    zz = [
        G.ZZ("angle").on(0), lambda angle:
        (-1j * angle * qt.tensor(qt.sigmaz(), qt.sigmaz())).expm()
    ]
    for g in [xx, yy, zz]:
        assert np.allclose(g[0].matrix({'angle': angle}),
                           g[1](angle).data.toarray())
        assert np.allclose(g[0].diff_matrix({'angle': angle}),
                           g[1](angle + np.pi / 2).data.toarray())


def test_pauli_gate():
    gates = {
        'X': [G.X, qt.sigmax(), qt.qip.operations.rx],
        'Y': [G.Y, qt.sigmay(), qt.qip.operations.ry],
        'Z': [G.Z, qt.sigmaz(), qt.qip.operations.rz]
    }
    angle = 0.5
    for name, ps in gates.items():
        assert np.allclose(ps[0].matrix(), ps[1].data.toarray())
        assert np.allclose((ps[0]**angle).matrix(),
                           ps[2](angle * np.pi).data.toarray())


def test_identity():
    assert np.allclose(G.I.matrix(), qt.identity(2).data.toarray())


def test_hadamard():
    assert np.allclose(G.H.matrix(),
                       qt.qip.operations.hadamard_transform(1).data.toarray())


def test_power():
    angle = 0.3
    frac = 0.4
    assert np.allclose(
        G.Power(G.RX(angle), frac).matrix(),
        G.RX(angle * frac).matrix())


def test_swap():
    assert np.allclose(G.SWAP.matrix(),
                       qt.qip.operations.swap().data.toarray())


def test_univ_mat_gate():
    mat = np.random.uniform(size=(2, 2))
    assert np.allclose(G.UnivMathGate('univ', mat).matrix(), mat)
