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

import numpy as np


def _check_num_array(vec, name):
    if not isinstance(vec, (np.ndarray, list)):
        raise TypeError(
            "{} requires a numpy.ndarray or a list of number, but get {}.".
            format(name, type(vec)))


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
        raise TypeError(
            "shape requires a int of a tuple of int, but get {}!".format(
                type(shapes)))
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


def _common_exp(num, tol=1e-7):
    """common expressions."""
    if num == 0:
        return num
    s2 = np.sqrt(2)
    s3 = np.sqrt(3)
    s5 = np.sqrt(5)
    com = {2: s2, 3: s3, 5: s5}
    for i, j in com.items():
        tmp_num = (j / num)
        ceil = np.ceil(tmp_num)
        floor = np.floor(tmp_num)
        if np.abs(tmp_num - ceil) < tol or np.abs(tmp_num - floor) < tol:
            frac = int(1 / (num / j))
            if frac > 0:
                return f'√{i}/{frac}'
            return f'-√{i}/{-frac}'
    return num


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
        >>> print('\\n'.join(ket_string(state)))
        √2/2¦0⟩
        -√2/2j¦1⟩
    """
    n = int(np.log2(len(state)))
    if len(state) < 2 and len(state) != (1 << n):
        raise ValueError("Invalid state size!")
    s = []
    for index, i in enumerate(state):
        b = _index_to_bitstring(index, n)
        if np.abs(i) < tol:
            continue
        if np.abs(np.real(i)) < tol:
            s.append(f'{_common_exp(np.imag(i), tol)}j¦{b}⟩')
            continue
        if np.abs(np.imag(i)) < tol:
            s.append(f'{_common_exp(np.real(i), tol)}¦{b}⟩')
            continue
        s.append(f'{i}¦{b}⟩')
    return s
