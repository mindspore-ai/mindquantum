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

import pytest
from mindquantum.ops import QubitOperator
import numpy as np
from scipy.linalg import expm
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
        'rx': [
            G.RX('angle').on(0), lambda phi: np.array([[
                np.cos(phi / 2), -1j * np.sin(phi / 2)
            ], [-1j * np.sin(phi / 2), np.cos(phi / 2)]])
        ],
        'ry': [
            G.RY('angle').on(0),
            lambda phi: np.array([[np.cos(phi / 2), -np.sin(
                phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        ],
        'rz': [
            G.RZ('angle').on(0),
            lambda phi: np.array([[np.exp(-1j * phi / 2), 0],
                                  [0, np.exp(1j * phi / 2)]])
        ]
    }
    angle = 0.5
    for name, rs in gates.items():
        assert np.allclose(rs[0].matrix({'angle': angle}),
                           rs[1](angle))
        assert np.allclose(rs[0].diff_matrix({'angle': angle}),
                           0.5 * rs[1](angle + np.pi))
        assert np.allclose(rs[0].hermitian().matrix({'angle': angle}),
                           rs[1](-angle))
        assert np.allclose(rs[0].hermitian().diff_matrix({'angle': angle}),
                           0.5 * rs[1](-angle - np.pi))


def test_phase_shift():
    angle = 0.5
    f = lambda theta: np.array([[1, 0], [0, np.exp(1.0j * theta)]])
    assert np.allclose(G.PhaseShift(angle).matrix(), f(angle))


def test_trap_ion_gate():
    angle = 0.5
    xx = [
        G.XX("angle").on((0, 1)), lambda angle: expm(-1j * angle * np.array(
            [[0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
             [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
             [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
             [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]))
    ]
    yy = [
        G.YY("angle").on((0, 1)), lambda angle: expm(-1j * angle * np.array(
            [[0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j],
             [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
             [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
             [-1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]))
    ]
    zz = [
        G.ZZ("angle").on((0, 1)), lambda angle: expm(-1j * angle * np.array([[
            1., 0., 0., 0.
        ], [0., -1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]]))
    ]
    for g in [xx, yy, zz]:
        assert np.allclose(g[0].matrix({'angle': angle}),
                           g[1](angle))
        assert np.allclose(g[0].diff_matrix({'angle': angle}),
                           g[1](angle + np.pi / 2))


def test_pauli_gate():
    gates = {
        'X': [
            G.X,
            np.array([[0. + 0.j, 1. + 0.j], [1. + 0.j, 0. + 0.j]]),
            lambda phi: np.array([[np.cos(phi / 2), -1j * np.sin(phi / 2)],
                                  [-1j * np.sin(phi / 2),
                                   np.cos(phi / 2)]])
        ],
        'Y': [
            G.Y,
            np.array([[0. + 0.j, 0. - 1.j], [0. + 1.j, 0. + 0.j]]),
            lambda phi: np.array([[np.cos(phi / 2), -np.sin(
                phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        ],
        'Z': [
            G.Z,
            np.array([[1. + 0.j, 0. + 0.j], [0. + 0.j, -1. + 0.j]]),
            lambda phi: np.array([[np.exp(-1j * phi / 2), 0],
                                  [0, np.exp(1j * phi / 2)]])
        ]
    }
    angle = 0.5
    for name, ps in gates.items():
        assert np.allclose(ps[0].matrix(), ps[1])
        assert np.allclose((ps[0]**angle).matrix(),
                           ps[2](angle * np.pi))


def test_identity():
    assert np.allclose(G.I.matrix(),
                       np.array([[1. + 0.j, 0. + 0.j], [0. + 0.j, 1. + 0.j]]))


def test_hadamard():
    assert np.allclose(
        G.H.matrix(),
        np.array([[0.70710678 + 0.j, 0.70710678 + 0.j],
                  [0.70710678 + 0.j, -0.70710678 + 0.j]]))


def test_power():
    angle = 0.3
    frac = 0.4
    assert np.allclose(
        G.Power(G.RX(angle), frac).matrix(),
        G.RX(angle * frac).matrix())


def test_swap():
    assert np.allclose(
        G.SWAP.matrix(),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]))


def test_univ_mat_gate():
    mat = np.random.uniform(size=(2, 2))
    assert np.allclose(G.UnivMathGate('univ', mat).matrix(), mat)


def test_gate_obj_mismatch():
    with pytest.raises(Exception, match=r"requires \d+ qubits"):
        G.X((0, 1))
    with pytest.raises(Exception, match=r"requires \d+ qubits"):
        G.RX('a').on((1, 2), 0)
    with pytest.raises(Exception, match=r"requires \d+ qubits"):
        G.RX(1).on((1, 2), 0)
    with pytest.raises(Exception, match=r"requires \d+ qubits"):
        G.ZZ('a').on(1, 0)


def test_gate_obj_ctrl_overlap():
    with pytest.raises(Exception, match=r"cannot have same qubits"):
        G.X(1, 1)
    with pytest.raises(Exception, match=r"cannot have same qubits"):
        G.ZZ('a').on((0, 1), (1, 2))
    with pytest.raises(Exception, match=r"cannot have same qubits"):
        G.RX('a').on(1, (1, 2))
