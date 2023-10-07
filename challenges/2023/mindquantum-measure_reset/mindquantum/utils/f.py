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

"""Useful functions."""

import numbers
from functools import lru_cache

import numpy as np

from mindquantum.config.config import _GLOBAL_MAT_VALUE

from .type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_between_close_set,
    _check_value_should_not_less,
)

__all__ = ['random_circuit', 'mod', 'normalize', 'random_state']


def random_circuit(n_qubits, gate_num, sd_rate=0.5, ctrl_rate=0.2, seed=None):
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
    # pylint: disable=import-outside-toplevel,cyclic-import
    from ..core import gates
    from ..core.circuit import Circuit

    _check_int_type('n_qubits', n_qubits)
    _check_int_type('gate_num', gate_num)
    _check_input_type('sd_rate', float, sd_rate)
    _check_input_type('ctrl_rate', float, ctrl_rate)
    if seed is None:
        seed = np.random.randint(1, 2**23)
    _check_int_type('seed', seed)
    _check_value_should_not_less('n_qubits', 1, n_qubits)
    _check_value_should_not_less('gate_num', 1, gate_num)
    _check_value_should_between_close_set('sd_rate', 0, 1, sd_rate)
    _check_value_should_between_close_set('ctrl_rate', 0, 1, ctrl_rate)
    _check_value_should_between_close_set('seed', 0, 2**32 - 1, seed)
    if n_qubits == 1:
        sd_rate = 1
        ctrl_rate = 0
    single = {
        'param': [gates.RX, gates.RY, gates.RZ, gates.PhaseShift],
        'non_param': [gates.X, gates.Y, gates.Z, gates.H],
    }
    double = {'param': [gates.Rxx, gates.Ryy, gates.Rzz], 'non_param': [gates.SWAP]}
    circuit = Circuit()
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
                param = np.random.uniform(-np.pi * 2, np.pi * 2)
                circuit += gate(param).on(q1, q2)
            else:
                gate = np.random.choice(single['non_param'])
                circuit += gate.on(q1, q2)
        else:
            if np.random.random() < 0.75:
                gate = np.random.choice(double['param'])
                param = np.random.uniform(-np.pi * 2, np.pi * 2)
                circuit += gate(param).on([q1, q2])
            else:
                gate = np.random.choice(double['non_param'])
                circuit += gate.on([q1, q2])
    return circuit


def _check_num_array(vec, name):
    if not isinstance(vec, (np.ndarray, list)):
        raise TypeError(f"{name} requires a numpy.ndarray or a list of number, but get {type(vec)}.")


def mod(vec_in, axis=0):
    """
    Calculate the mod of input vectors.

    Args:
        vec_in (Union[list[numbers.Number], numpy.ndarray]): The vector you want to calculate mod.
        axis (int): Along which axis you want to calculate mod. Default: ``0``.

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
        axis (int): Along which axis you want to normalize your vector. Default: ``0``.

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
        norm_axis (int): which axis you want to apply normalization. Default: ``0``.
        comp (bool): if ``True``, each amplitude of the quantum state will be a
            complex number. Default: ``True``.
        seed (int): the random seed. Default: ``None``.

    Returns:
        numpy.ndarray, A normalized random quantum state.

    Examples:
        >>> from mindquantum.utils import random_state
        >>> random_state((2, 2), seed=42)
        array([[0.44644744+0.18597239j, 0.66614846+0.10930256j],
               [0.87252821+0.06923499j, 0.41946926+0.60691409j]])
    """
    if not isinstance(shapes, (int, tuple)):
        raise TypeError(f"shape requires a int of a tuple of int, but get {type(shapes)}!")
    if not isinstance(comp, bool):
        raise TypeError(f"comp requires a bool, but get {comp}!")
    np.random.seed(seed)
    out = np.random.uniform(size=shapes) + 0j
    if comp:
        out += np.random.uniform(size=shapes) * 1j
    if norm_axis is False:
        return out
    return normalize(out, axis=norm_axis)


def is_two_number_close(a, b, atol=None):  # pylint: disable=invalid-name
    """
    Check whether two number is close within the error of atol.

    This method also works for complex numbers.

    Args:
        a (numbers.Number): The first number.
        b (numbers.Number): The second number.
        atol (float): The atol. If ``None``, the precision defined in global config
            will be used. Default: ``None``.

    Returns:
        bool, whether this two number close to each other.

    Examples:
        >>> from mindquantum.utils import is_two_number_close
        >>> is_two_number_close(1+1j, 1+1j)
        True
    """
    from mindquantum.config.config import (  # pylint: disable=import-outside-toplevel
        context,
    )

    _check_input_type("a", numbers.Number, a)
    _check_input_type("b", numbers.Number, b)
    if atol is None:
        atol = context.get_precision()
    _check_input_type("atol", float, atol)
    return np.allclose(np.abs(a - b), 0, atol=atol)


def is_power_of_two(num):
    """Check whether a number is power of 2 or not."""
    return (num & (num - 1) == 0) and num != 0


@lru_cache()
def pauli_string_matrix(pauli_string):
    """
    Generate the matrix of pauli string.

    If pauli string is XYZ, then the matrix will be `Z@Y@X`.
    """
    try:
        matrix = _GLOBAL_MAT_VALUE[pauli_string[0]]
        for string in pauli_string[1:]:
            matrix = np.kron(_GLOBAL_MAT_VALUE[string], matrix)
    except KeyError as err:
        raise err
    return matrix
