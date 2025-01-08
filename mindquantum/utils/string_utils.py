#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""String helper functions."""

import fractions
import numbers

import numpy as np

from .f import is_two_number_close
from .type_value_check import _check_input_type


def join_without_empty(joiner, string):
    """Join strings and skip empty string."""
    return joiner.join(filter(lambda x: x and x.strip(), string))


def _index_to_bitstring(index, n, big_end=False):
    """Transfer the index to bitstring."""
    bitstring = bin(index)[2:].zfill(n)
    if big_end:
        return bitstring[::-1]
    return bitstring


def real_string_expression(num, round_n=None):
    """
    Convert a real number to string expression.

    Returns:
        str, the string expression of given real number.
    """
    _check_input_type('num', numbers.Real, num)
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


def string_expression(arg):
    """
    Convert a number, complex number included, to string expression.

    Returns:
        str, the string expression of given number.
    """
    _check_input_type('x', numbers.Number, arg)
    real_x = np.real(arg)
    imag_x = np.imag(arg)
    if is_two_number_close(arg, 0):
        return '0'
    if is_two_number_close(imag_x, 0):
        res = real_string_expression(real_x)
        if res == str(real_x):
            return str(np.round(real_x, 4))
        return res
    if is_two_number_close(real_x, 0):
        return string_expression(imag_x) + 'j'
    real_part = string_expression(real_x)
    imag_part = string_expression(imag_x * 1j)
    if imag_part.startswith('-'):
        return real_part + imag_part
    return real_part + ' + ' + imag_part


def ket_string(state, tol=1e-7):
    """
    Get the ket format of the quantum state.

    Args:
        state (numpy.ndarray): The input quantum state.
        tol (float): The ignore tolerance for small amplitude. Default: ``1e-7``.

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
        raise TypeError("state need a 1-D ndarray.")
    n = int(np.log2(len(state)))
    if len(state) < 2 and len(state) != (1 << n):
        raise ValueError("Invalid state size!")
    string = []
    for index, i in enumerate(state):
        bitstring = _index_to_bitstring(index, n)
        if np.abs(i) < tol:
            continue
        if np.abs(np.real(i)) < tol:
            string.append(f'{real_string_expression(np.imag(i), round_n=8)}j¦{bitstring}⟩')
            continue
        if np.abs(np.imag(i)) < tol:
            string.append(f'{real_string_expression(np.real(i), round_n=8)}¦{bitstring}⟩')
            continue
        i_real = real_string_expression(np.real(i), round_n=8)
        i_imag = real_string_expression(np.imag(i), round_n=8)
        if i_imag.startswith('-'):
            string.append(f'({i_real}{i_imag}j)¦{bitstring}⟩')
        else:
            string.append(f'({i_real}+{i_imag}j)¦{bitstring}⟩')
    return string


def density_matrix_ket_string(density_matrix, indent='    '):
    """
    Display density matrix in a ket format.

    Args:
        density_matrix (numpy.ndarray): Density matrix for n qubits
        indent (str): Indentation for multiline display. Default: 4 spaces

    Returns:
        str: the ket format of density matrix
    """
    purity = np.real(np.trace(density_matrix @ density_matrix))

    if abs(purity - 1.0) < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
        max_idx = np.argmax(eigenvalues)
        state_vector = eigenvectors[:, max_idx]

        first_nonzero = np.nonzero(state_vector)[0][0]
        if state_vector[first_nonzero].real < 0:
            state_vector = -state_vector

        components = ket_string(state_vector)
        if len(components) > 3:
            return (' +\n').join(components)
        return ' + '.join(components)

    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    terms = []
    for prob, state_vector in zip(eigenvalues, eigenvectors.T):
        if abs(prob) > 1e-6:
            components = ket_string(state_vector)
            prob_str = real_string_expression(prob, round_n=8)
            components = [c[1:] if c.startswith('1¦') else c for c in components]

            total_len = sum(len(c) for c in components)
            if len(components) > 3 or total_len > 50:
                state_str = indent + (' +\n' + indent).join(components)
                terms.append(f"{prob_str}(\n{state_str}\n)")
            else:
                state_str = ' + '.join(components)
                if "+" in state_str:
                    state_str = f"({state_str})"
                terms.append(f"{prob_str}{state_str}")

    if len(terms) > 1:
        total_terms_len = sum(len(t) for t in terms)
        if total_terms_len > 50:
            return " +\n".join(terms) + " (mixed state)"
        return " + ".join(terms) + " (mixed state)"
    return terms[0] + " (mixed state)"
