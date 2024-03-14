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
"""Qudit symmetric mapping module."""

import numpy as np
from scipy.sparse import csr_matrix
from mindquantum.utils.f import is_power_of_two
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RX, RY, RZ, U3, GlobalPhase, UnivMathGate

optional_basis = ['zyz', 'u3']


def _symmetric_index(dim: int, n_qudits: int) -> dict:
    '''Index of symmetric mapping.'''
    if not isinstance(dim, int):
        raise ValueError(f'Wrong dimension type {dim} {type(dim)}')
    if not isinstance(n_qudits, int):
        raise ValueError(f'Wrong number of qudits type {n_qudits} {type(n_qudits)}')
    if n_qudits == 1:
        ind = {}
        for i in range(2**(dim - 1)):
            num = bin(i).count('1')
            if num in ind:
                ind[num].append(i)
            else:
                ind[num] = [i]
    else:
        ind, ind_ = {}, {}
        for i in range(2**(dim - 1)):
            num = bin(i).count('1')
            i_ = bin(i)[2::].zfill(dim - 1)
            if num in ind_:
                ind_[num].append(i_)
            else:
                ind_[num] = [i_]
        for i in range(dim**n_qudits):
            multi = ['']
            base = np.base_repr(i, dim).zfill(n_qudits)
            for j in range(n_qudits):
                multi = [x + y for x in multi for y in ind_[int(base[j])]]
            ind[i] = [int(x, 2) for x in multi]
    return ind


def qudit_symmetric_decoding(qubit: np.ndarray, n_qudits: int = 1) -> np.ndarray:
    '''Qudit symmetric decoding.'''
    if qubit.ndim == 2 and (qubit.shape[0] == 1 or qubit.shape[1] == 1):
        qubit = qubit.flatten()
    if qubit.ndim == 2 and qubit.shape[0] != qubit.shape[1]:
        raise ValueError(f'Wrong qubit shape {qubit.shape}')
    if qubit.ndim != 1 and qubit.ndim != 2:
        raise ValueError(f'Wrong qubit shape {qubit.shape}')
    n = qubit.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f'Wrong matrix size {n} is not a power of 2')
    nq = int(np.log2(n))
    dim = nq // n_qudits + 1
    if nq % n_qudits == 0 and nq != n_qudits:
        ind = _symmetric_index(dim, n_qudits)
    else:
        raise ValueError(f'Wrong matrix shape {qubit.shape} or number of qudits {n_qudits}')
    if qubit.ndim == 1:
        qudit = np.zeros(dim**n_qudits, dtype=np.complex128)
        for i in range(dim**n_qudits):
            i_ = ind[i]
            qubit_i = qubit[i_]
            if np.allclose(qubit_i, qubit_i[0]):
                qudit[i] = qubit_i[0] * np.sqrt(len(i_))
            else:
                raise ValueError('Qubit matrix is not symmetric')
    elif qubit.ndim == 2:
        qudit = np.zeros([dim**n_qudits, dim**n_qudits], dtype=np.complex128)
        for i in range(dim**n_qudits):
            i_ = ind[i]
            for j in range(dim**n_qudits):
                j_ = ind[j]
                qubit_ij = qubit[np.ix_(i_, j_)]
                if np.allclose(qubit_ij, qubit_ij[0][0]):
                    div = np.sqrt(len(i_)) * np.sqrt(len(j_))
                    qudit[i, j] = qubit_ij[0][0] * div
                else:
                    raise ValueError('Qubit matrix is not symmetric')
    return qudit


def qudit_symmetric_encoding(qudit: np.ndarray, n_qudits: int = 1, is_csr: bool = False) -> np.ndarray:
    '''Qudit symmetric encoding.'''
    if qudit.ndim == 2 and (qudit.shape[0] == 1 or qudit.shape[1] == 1):
        qudit = qudit.flatten()
    if qudit.ndim == 2 and qudit.shape[0] != qudit.shape[1]:
        raise ValueError(f'Wrong qudit shape {qudit.shape}')
    if qudit.ndim != 1 and qudit.ndim != 2:
        raise ValueError(f'Wrong qudit shape {qudit.shape}')
    dim = round(qudit.shape[0]**(1 / n_qudits), 12)
    if dim % 1 == 0:
        dim = int(dim)
        n = 2**((dim - 1) * n_qudits)
        ind = _symmetric_index(dim, n_qudits)
    else:
        raise ValueError(f'Wrong qudit shape {qudit.shape} or number of qudits {n_qudits}')
    if qudit.ndim == 1:
        qubit = csr_matrix((n, 1), dtype=np.complex128)
        for i in range(dim**n_qudits):
            ind_i = ind[i]
            num_i = len(ind_i)
            data = np.ones(num_i) * qudit[i] / np.sqrt(num_i)
            i_ = (ind_i, np.zeros(num_i))
            qubit += csr_matrix((data, i_), shape=(n, 1))
        if not is_csr:
            qubit = qubit.toarray().flatten()
    elif qudit.ndim == 2:
        qubit = csr_matrix((n, n), dtype=np.complex128)
        for i in range(dim**n_qudits):
            ind_i = ind[i]
            num_i = len(ind_i)
            for j in range(dim**n_qudits):
                ind_j = ind[j]
                num_j = len(ind_j)
                i_ = np.repeat(ind_i, num_j)
                j_ = np.tile(ind_j, num_i)
                div = np.sqrt(num_i) * np.sqrt(num_j)
                data = np.ones(num_i * num_j) * qudit[i, j] / div
                qubit += csr_matrix((data, (i_, j_)), shape=(n, n))
        if not is_csr:
            qubit = qubit.toarray()
    return qubit


def _two_level_unitary_synthesize(basis: str, dim: int, ind: list, pr: list, obj: list) -> Circuit:
    '''Synthesize a two-level unitary qudit gate with qubit circuit.'''
    if dim != 3:
        raise ValueError('Currently only works when dim = 3')
    if len(ind) != 2:
        raise ValueError(f'U3 index length {len(ind)} should be 2')
    if len(set(ind)) != len(ind):
        raise ValueError(f'U3 index {ind} cannot be repeated')
    if min(ind) < 0 or max(ind) >= dim:
        raise ValueError(f'U3 index {ind} should in 0 to {dim-1}')
    if len(pr) != 3:
        raise ValueError(f'U3 params length {len(pr)} should be 3')
    circ = Circuit()
    if ind == [0, 1]:
        corr = Circuit() + X(obj[1], obj[0]) + RY(np.pi / 2).on(obj[0], obj[1]) + X(obj[1], obj[0]) + X(obj[1])
    elif ind == [0, 2]:
        corr = Circuit() + X(obj[0]) + X(obj[1], obj[0]) + X(obj[0])
    elif ind == [1, 2]:
        corr = Circuit() + X(obj[1], obj[0]) + RY(-np.pi / 2).on(obj[0], obj[1]) + X(obj[1], obj[0])
    circ += corr
    if basis == 'zyz':
        circ += RZ(pr[0]).on(obj[0], obj[1])
        circ += RY(pr[1]).on(obj[0], obj[1])
        circ += RZ(pr[2]).on(obj[0], obj[1])
    elif basis == 'u3':
        theta, phi, lam = pr
        circ += U3(theta, phi, lam).on(obj[0], obj[1])
    else:
        raise ValueError(f'Wrong basis {basis} is not in {optional_basis}')
    circ += corr.hermitian()
    return circ


def _single_qudit_unitary_synthesize(basis: str, dim: int, name: str, obj: list) -> Circuit:
    '''Synthesize a single qudit unitary gate with qubit circuit.'''
    circ = Circuit()
    index = [[0, 1], [0, 2], [1, 2]]
    if basis == 'zyz':
        for i, ind in enumerate(index):
            str_pr = f'{"".join(str(i) for i in ind)}_{i}'
            pr = [f'{name}RZ{str_pr}', f'{name}RY{str_pr}', f'{name}Rz{str_pr}']
            circ += _two_level_unitary_synthesize(basis, dim, ind, pr, obj)
    elif basis == 'u3':
        for i, ind in enumerate(index):
            str_pr = f'{"".join(str(i) for i in ind)}_{i}'
            pr = [f'{name}𝜃{str_pr}', f'{name}𝜑{str_pr}', f'{name}𝜆{str_pr}']
            circ += _two_level_unitary_synthesize(basis, dim, ind, pr, obj)
    else:
        raise ValueError(f'Wrong basis {basis} is not in {optional_basis}')
    return circ


def _controlled_rotation_synthesize(dim: int, ind: list, name: str, obj: int, ctrl: list, state: int) -> Circuit:
    '''Synthesize a controlled rotation qudit gate with qubit circuit.'''
    if dim != 3:
        raise ValueError('Currently only works when dim = 3')
    circ = Circuit()
    if state == 0:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(obj, ctrl[1:]) + X(ctrl[0], ctrl[1:] + [obj]) + X(
                obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(-np.pi / 2).on(
                obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj])
    elif state == 1:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(
                np.pi / 2).on(obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(obj, ctrl[1:]) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(
                -np.pi / 2).on(obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj])
    elif state == 2:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[0], ctrl[1:] + [obj]) + RY(np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(obj, ctrl[1:]) + X(ctrl[0], ctrl[1:] + [obj]) + X(obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[0], ctrl[1:] + [obj]) + RY(-np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj])
    circ += corr
    if 'RX' in name:
        circ = circ + RX(name).on(obj, ctrl)
    elif 'RY' in name:
        circ = circ + RY(name).on(obj, ctrl)
    elif 'RZ' in name:
        circ = circ + RZ(name).on(obj, ctrl)
    elif 'GP' in name:
        circ = circ + GlobalPhase(name).on(obj, ctrl)
    circ += corr.hermitian()
    return circ


def _controlled_diagonal_synthesize(dim: int, name: str, obj: int, ctrl: list, state: int) -> Circuit:
    '''Synthesize a controlled diagonal qudit gate with qubit circuit.'''
    if dim != 3:
        raise ValueError('Currently only works when dim = 3')
    circ = Circuit()
    circ += _controlled_rotation_synthesize(dim, [0, 1], f'{name}RZ01', obj, ctrl, state)
    circ += _controlled_rotation_synthesize(dim, [0, 2], f'{name}RZ02', obj, ctrl, state)
    circ += _controlled_rotation_synthesize(dim, [0, 1], f'{name}GP', obj, ctrl, state)
    circ += _controlled_rotation_synthesize(dim, [0, 2], f'{name}GP', obj, ctrl, state)
    circ += _controlled_rotation_synthesize(dim, [1, 2], f'{name}GP', obj, ctrl, state)
    return circ


def qutrit_symmetric_ansatz(gate: UnivMathGate, basis: str = 'zyz', with_phase: bool = False) -> Circuit:
    """Qudit symmetric ansatz."""
    dim = 3
    name = f'{gate.name}_'
    obj = gate.obj_qubits
    circ = Circuit()
    if len(obj) == 2:
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}', obj)
    elif len(obj) == 4:
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U1_', obj[:2])
        circ += _controlled_diagonal_synthesize(dim, f'{name}C1_', obj[0], obj[1:], 1)
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U2_', obj[:2])
        circ += _controlled_diagonal_synthesize(dim, f'{name}C2_', obj[0], obj[1:], 2)
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U3_', obj[:2])
        circ += _controlled_rotation_synthesize(dim, [1, 2], f'{name}RY12', obj[-1], obj[::-1][1:], 2)
        circ += _controlled_rotation_synthesize(dim, [1, 2], f'{name}RY11', obj[-1], obj[::-1][1:], 1)
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U4_', obj[:2])
        circ += _controlled_diagonal_synthesize(dim, f'{name}C3_', obj[0], obj[1:], 2)
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U5_', obj[:2])
        circ += _controlled_rotation_synthesize(dim, [0, 1], f'{name}RY22', obj[-1], obj[::-1][1:], 2)
        circ += _controlled_rotation_synthesize(dim, [0, 1], f'{name}RY21', obj[-1], obj[::-1][1:], 1)
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U6_', obj[:2])
        circ += _controlled_diagonal_synthesize(dim, f'{name}C4_', obj[0], obj[1:], 0)
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U7_', obj[:2])
        circ += _controlled_rotation_synthesize(dim, [1, 2], f'{name}RY32', obj[-1], obj[::-1][1:], 2)
        circ += _controlled_rotation_synthesize(dim, [1, 2], f'{name}RY31', obj[-1], obj[::-1][1:], 1)
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U8_', obj[:2])
        circ += _controlled_diagonal_synthesize(dim, f'{name}C5_', obj[0], obj[1:], 2)
        circ += _single_qudit_unitary_synthesize(basis, dim, f'{name}U9_', obj[:2])
    else:
        raise ValueError('Only works when number of qutrits <= 2')
    if with_phase:
        for i in obj:
            circ += GlobalPhase(f'{name}phase').on(i)
    return circ
