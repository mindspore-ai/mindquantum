# -*- coding: utf-8 -*-
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
"""Useful functions"""

import fractions
import numpy as np
from .type_value_check import _check_input_type
from .type_value_check import _check_int_type
from .type_value_check import _check_value_should_not_less
from .type_value_check import _check_value_should_between_close_set


def random_circuit(n_qubits, gate_num, sd_rate=0.5, ctrl_rate=0.2, seed=42):
    """
    Generate a random circuit.

    Args:
        n_qubits (int): Number of qubits of random circuit.
        gate_num (int): Number of gates in random circuit.
        sd_rate (float): The rate of single qubit gate and double qubits gates.
        ctrl_rate (float): The possibility that a gate has a control qubit.
        seed (int): Random seed to generate random circuit.

    Examples:
        >>> from mindquantum.utils import random_circuit
        >>> random_circuit(3, 4, 0.5, 0.5, 100)
        q1: ──Z────RX(0.944)────────●────────RX(-0.858)──
              │        │            │            │
        q2: ──●────────●────────RZ(-2.42)────────●───────
    """
    from mindquantum import Circuit
    import mindquantum.core.gates as G
    _check_int_type('n_qubits', n_qubits)
    _check_int_type('gate_num', gate_num)
    _check_input_type('sd_rate', float, sd_rate)
    _check_input_type('ctrl_rate', float, ctrl_rate)
    _check_int_type('seed', seed)
    _check_value_should_not_less('n_qubits', 1, n_qubits)
    _check_value_should_not_less('gate_num', 1, gate_num)
    _check_value_should_between_close_set('sd_rate', 0, 1, sd_rate)
    _check_value_should_between_close_set('ctrl_rate', 0, 1, ctrl_rate)
    _check_value_should_between_close_set('seed', 0, 2**32 - 1, seed)
    if n_qubits == 1:
        sd_rate = 1
        ctrl_rate = 0
    single = {'param': [G.RX, G.RY, G.RZ, G.PhaseShift], 'non_param': [G.X, G.Y, G.Z, G.H]}
    double = {'param': [G.XX, G.YY, G.ZZ], 'non_param': [G.SWAP]}
    c = Circuit()
    np.random.seed(seed)
    qubits = range(n_qubits)
    for _ in range(gate_num):
        if n_qubits == 1:
            q1, q2 = int(qubits[0]), None
        else:
            q1, q2 = np.random.choice(qubits, 2, replace=False)
            q1, q2 = int(q1), int(q2)
        if np.random.random() < sd_rate:
            if np.random.random() > ctrl_rate:
                q2 = None
            if np.random.random() < 0.5:
                gate = np.random.choice(single['param'])
                p = np.random.uniform(-np.pi * 2, np.pi * 2)
                c += gate(p).on(q1, q2)
            else:
                gate = np.random.choice(single['non_param'])
                c += gate.on(q1, q2)
        else:
            if np.random.random() < 0.75:
                gate = np.random.choice(double['param'])
                p = np.random.uniform(-np.pi * 2, np.pi * 2)
                c += gate(p).on([q1, q2])
            else:
                gate = np.random.choice(double['non_param'])
                c += gate.on([q1, q2])
    return c


def _check_num_array(vec, name):
    if not isinstance(vec, (np.ndarray, list)):
        raise TypeError("{} requires a numpy.ndarray or a list of number, but get {}.".format(name, type(vec)))


def mod(vec_in, axis=0):
    """
    Calculate the mod of input vectors.

    Args:
        vec_in (Union[list[numbers.Number], numpy.ndarray]): The vector you want to calculate mod.
        axis (int): Along which axis you want to calculate mod. Default: 0.

    Returns:
        numpy.ndarray, The mod of input vector.

    Examples:
        >>> from mindquantum.utils import mod
        >>> vec_in = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mod(vec_in)
        array([[4.12310563, 5.38516481, 6.70820393]])
        >>> mod(vec_in, 1)
        array([[3.74165739],
               [8.77496439]])
    """
    _check_num_array(vec_in, 'vec_in')
    vec_in = np.array(vec_in)
    return np.sqrt(np.sum(np.conj(vec_in) * vec_in, axis=axis, keepdims=True))


def normalize(vec_in, axis=0):
    """
    Normalize the input vectors based on specified axis.

    Args:
        vec_in (Union[list[number], numpy.ndarray]): Vector you want to
            normalize.
        axis (int): Along which axis you want to normalize your vector. Default: 0.

    Returns:
        numpy.ndarray, Vector after normalization.

    Examples:
        >>> from mindquantum.utils import normalize
        >>> vec_in = np.array([[1, 2, 3], [4, 5, 6]])
        >>> normalize(vec_in)
        array([[0.24253563, 0.37139068, 0.4472136 ],
               [0.9701425 , 0.92847669, 0.89442719]])
        >>> normalize(vec_in, 1)
        array([[0.26726124, 0.53452248, 0.80178373],
               [0.45584231, 0.56980288, 0.68376346]])
    """
    _check_num_array(vec_in, 'vec_in')
    vec_in = np.array(vec_in)
    return vec_in / mod(vec_in, axis=axis)


def random_state(shapes, norm_axis=0, comp=True, seed=None):
    r"""
    Generate some random quantum state.

    Args:
        shapes (tuple): shapes = (m, n) means m quantum states with each state
            formed by :math:`\log_2(n)` qubits.
        norm_axis (int): which axis you want to apply normalization. Default: 0.
        comp (bool): if `True`, each amplitude of the quantum state will be a
            complex number. Default: True.
        seed (int): the random seed. Default: None.

    Returns:
        numpy.ndarray, A normalized random quantum state.

    Examples:
        >>> from mindquantum.utils import random_state
        >>> random_state((2, 2), seed=42)
        array([[0.44644744+0.18597239j, 0.66614846+0.10930256j],
               [0.87252821+0.06923499j, 0.41946926+0.60691409j]])
    """
    if not isinstance(shapes, (int, tuple)):
        raise TypeError("shape requires a int of a tuple of int, but get {}!".format(type(shapes)))
    if not isinstance(comp, bool):
        raise TypeError("comp requires a bool, but get {}!".format(comp))
    np.random.seed(seed)
    out = np.random.uniform(size=shapes) + 0j
    if comp:
        out += np.random.uniform(size=shapes) * 1j
    if norm_axis is False:
        return out
    return normalize(out, axis=norm_axis)


def _index_to_bitstring(index, n, big_end=False):
    """Transfor the index to bitstring"""
    s = bin(index)[2:].zfill(n)
    if big_end:
        return s[::-1]
    return s


def _common_exp(num, round_n=None):
    """common expressions."""
    if num == 0:
        return '0'
    com = {'': 1, 'π': np.pi, '√2': np.sqrt(2), '√3': np.sqrt(3), '√5': np.sqrt(5)}
    for k, v in com.items():
        left = str(fractions.Fraction(str(round(num / v, 9))))
        if len(left) < 5 or '/' not in left or left.startswith('1/') or left.startswith('-1/'):
            tmp = left.split('/')
            if not (len(tmp) == 2 and int(tmp[1]) > 5 and int(tmp[0]) > 5):
                if tmp[0] == '1':
                    tmp[0] = k
                    if k == '':
                        tmp[0] = '1'
                elif tmp[0] == '-1':
                    tmp[0] = f"-{k}"
                    if k == '':
                        tmp[0] = '-1'
                else:
                    tmp[0] = f"{tmp[0]}{k}"
                return '/'.join(tmp)
    return str(num) if round_n is None else str(round(num, round_n))


def ket_string(state, tol=1e-7):
    """
    Get the ket format of the quantum state.

    Args:
        state (numpy.ndarray): The input quantum state.
        tol (float): The ignore tolence for small amplitude. Default: 1e-7.

    Returns:
        str, the ket format of the quantum state.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.utils import ket_string
        >>> state = np.array([1, -1j])/np.sqrt(2)
        >>> print(ket_string(state))
        ['√2/2¦0⟩', '-√2/2j¦1⟩']
    """
    if not isinstance(state, np.ndarray) or len(state.shape) != 1:
        raise TypeError(f"state need a 1-D ndarray.")
    n = int(np.log2(len(state)))
    if len(state) < 2 and len(state) != (1 << n):
        raise ValueError("Invalid state size!")
    s = []
    for index, i in enumerate(state):
        b = _index_to_bitstring(index, n)
        if np.abs(i) < tol:
            continue
        if np.abs(np.real(i)) < tol:
            s.append(f'{_common_exp(np.imag(i))}j¦{b}⟩')
            continue
        if np.abs(np.imag(i)) < tol:
            s.append(f'{_common_exp(np.real(i))}¦{b}⟩')
            continue
        i_real = _common_exp(np.real(i))
        i_imag = _common_exp(np.imag(i))
        if i_imag.startswith('-'):
            s.append(f'({i_real}{i_imag}j)¦{b}⟩')
        else:
            s.append(f'({i_real}+{i_imag}j)¦{b}⟩')
    return s
