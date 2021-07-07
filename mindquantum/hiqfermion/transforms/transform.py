#   Copyright (c) 2020 Huawei Technologies Co.,ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
This module is to transform the fermion type operator to qubit type operator,
thus can be simulated in quantum computer
"""
from math import floor, log
import numpy as np

from mindquantum.ops.fermion_operator import FermionOperator
from mindquantum.ops.qubit_operator import QubitOperator
from mindquantum.utils import count_qubits, number_operator, normal_ordered


class Transform:
    r"""
    Class for transforms of fermionic and qubit operators.
    Methods jordan_wigner, parity, bravyi_kitaev, bravyi_kitaev_tree,
    bravyi_kitaev_superfast make transform of fermionic operators to
    qubit ones,
    they are initialized by FermionOperator, return QubitOperator.
    Note method reversed_jordan_wigner makes transform of qubit operator
    to fermionic one, it is initialized by QubitOperator,
    returns FermionOperator.

    Args:
        operator (Union[FermionOperator, QubitOperator]): The input
            FermionOperator or QubitOperator that need to do transform.
        n_qubits (int): The total qubits of this operator. Default: None

    Examples:
        >>> from mindquantum.ops import FermionOperator
        >>> op1 = FermionOperator('1^')
        >>> op1
        1.0 [1^]
        >>> from mindquantum.hiqfermion.transforms.transform import Transform
        >>> op_transform = Transform(op1)
        >>> op_transform.jordan_wigner()
        0.5 [Z0 X1] +
        -0.5j [Z0 Y1]
        >>> op_transform.parity()
        0.5 [Z0 X1] +
        -0.5j [Y1]
        >>> op_transform.bravyi_kitaev()
        0.5 [Z0 X1] +
        -0.5j [Y1]
        >>> op_transform.ternary_tree()
        0.5 [X0 Z1] +
        -0.5j [Y0 X2]
        >>> op2 = FermionOperator('1^', 'a')
        >>> Transform(op2).jordan_wigner()
        0.5*a [Z0 X1] +
        -0.5*I*a [Z0 Y1]
    """
    def __init__(self, operator, n_qubits=None):
        if not isinstance(operator, (FermionOperator, QubitOperator)):
            raise TypeError(
                "Operator must be a FermionOperator or QubitOperator")
        if n_qubits is None:
            n_qubits = count_qubits(operator)
        if n_qubits < count_qubits(operator):
            raise ValueError('Invalid number of qubits specified.')

        self.n_qubits = n_qubits
        self.operator = operator

    def jordan_wigner(self):
        r"""
        Apply Jordan-Wigner transform. The Jordan-Wigner transform
        holds the initial occupation number locally.
        which change the formular of fermion
        operator into qubit operator following the equation.

        .. math::

            a^\dagger_{j}\rightarrow \sigma^{-}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i}

            a_{j}\rightarrow \sigma^{+}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i},

        where the :math:`\sigma_{+}= \sigma^{X} + i \sigma^{Y}` and :math:`\sigma_{-} = \sigma^{X} - i\sigma^{Y}` is the
        Pauli spin raising and lowring operator.

        Returns:
            QubitOperator, qubit operator after jordan_wigner transformation.
        """
        if not isinstance(self.operator, FermionOperator):
            raise TypeError(
                'This method can be only applied for FermionOperator.')
        transf_op = QubitOperator()
        for term in self.operator.terms:
            # Initialize identity matrix.
            transformed_term = QubitOperator((), self.operator.terms[term])

            # Loop through operators, transform and multiply.
            for ladder_operator in term:
                # Define lists of qubits to apply Pauli gates
                index = ladder_operator[0]
                x1 = [index]
                y1 = []
                z1 = list(range(index))
                x2 = []
                y2 = [index]
                z2 = list(range(index))
                transformed_term *= _transform_ladder_operator(
                    ladder_operator, x1, y1, z1, x2, y2, z2)
            transf_op += transformed_term

        return transf_op

    def parity(self):
        r"""
        Apply parity transform.
        The parity transform
        stores the initial occupation number nonlocally.
        with the formular:

        .. math::

            \left|f_{M−1}, f_{M−2},\cdots, f_0\right> → \left|q_{M−1}, q_{M−2},\cdots, q_0\right>,

        where

        .. math::

            q_{m} = \left|\left(\sum_{i=0}^{m-1}f_{i}\right) mod\ 2 \right>

        Basically, this formular could be written as this,

        .. math::

            p_{i} = \sum{[\pi_{n}]_{i,j}} f_{j},

        where :math:`\pi_{n}` is the :math:`N\times N` square matrix,
        :math:`N` is the total qubit number. The operator changes follows the following equation as:

        .. math::

            a^\dagger_{j}\rightarrow\frac{1}{2}\left(\prod_{i=j+1}^N
            \left(\sigma_i^X X\right)\right)\left( \sigma^{X}_{j}-i\sigma_j^Y\right) X \sigma^{Z}_{j-1}

            a_{j}\rightarrow\frac{1}{2}\left(\prod_{i=j+1}^N
            \left(\sigma_i^X X\right)\right)\left( \sigma^{X}_{j}+i\sigma_j^Y\right) X \sigma^{Z}_{j-1}

        Returns:
            QubitOperator, qubits operator after parity transformation.
        """
        if not isinstance(self.operator, FermionOperator):
            raise TypeError(
                'This method can be only applied for FermionOperator.')
        transf_op = QubitOperator()
        for term in self.operator.terms:
            # Initialize identity matrix.
            transformed_term = QubitOperator((), self.operator.terms[term])

            # Loop through operators, transform and multiply.
            for ladder_operator in term:
                # Define lists of qubits to apply Pauli gates
                index = ladder_operator[0]
                x1 = list(range(index, self.n_qubits))
                y1 = []
                z1 = [index - 1] if index > 0 else []
                x2 = list(range(index + 1, self.n_qubits))
                y2 = [index]
                z2 = []
                transformed_term *= _transform_ladder_operator(
                    ladder_operator, x1, y1, z1, x2, y2, z2)
            transf_op += transformed_term

        return transf_op

    def bravyi_kitaev(self):
        r"""
        Apply Bravyi-Kitaev transform.

        The Bravyi-Kitaev basis is a middle between Jordan-Wigner
        and parity transform. That is, it balances the locality of occupation and parity information
        for improved simulation efficiency. In this scheme, qubits store the parity
        of a set of :math:`2^x` orbitals, where :math:`x \ge 0`. A qubit of index j always
        stores orbital :math:`j`.
        For even values of :math:`j`, this is the only orbital
        that it stores, but for odd values of :math:`j`, it also stores a certain
        set of adjacent orbitals with index less than :math:`j`.
        For the occupation transformation, we follow the
        formular:

        .. math::

            b_{i} = \sum{[\beta_{n}]_{i,j}} f_{j},

        where :math:`\beta_{n}` is the :math:`N\times N` square matrix,
        :math:`N` is the total qubit number.
        The qubits index are divide into three sets,
        the parity set, the update set and flip set.
        The parity of this set of qubits has
        the same parity as the set of orbitals with index less than :math:`j`,
        and so we will call this set of qubit indices the "parity set" of
        index :math:`j`, or :math:`P(j)`.

        the update set of index :math:`j`, or :math:`U(j)` contains the set of qubits (other than
        qubit :math:`j`) that must be updated when the occupation of orbital :math:`j`
        This is the set of qubits in the Bravyi-Kitaev basis that store a
        partial sum including orbital :math:`j`.
        the flip set of index :math:`j`, or :math:`F(j)` contains the set of BravyiKitaev qubits determines
        whether qubit :math:`j` has the same parity or inverted parity with
        respect to orbital :math:`j`.

        Please see some detail explanation in the paper (THE JOURNAL OF
        CHEMICAL PHYSICS 137, 224109 (2012)).

        Implementation from https://arxiv.org/pdf/quant-ph/0003137.pdf and
        "A New Data Structure for Cumulative Frequency Tables"
        by Peter M. Fenwick.

        Returns:
            QubitOperator, qubit operator after bravyi_kitaev transformation.
        """
        if not isinstance(self.operator, FermionOperator):
            raise TypeError(
                'This method can be only applied for FermionOperator.')
        transf_op = QubitOperator()
        for term in self.operator.terms:
            # Initialize identity matrix.
            transformed_term = QubitOperator((), self.operator.terms[term])

            # Loop through operators, transform and multiply.
            for ladder_operator in term:
                # Define lists of qubits to apply Pauli gates
                index = ladder_operator[0]
                update_set = _update_set(index, self.n_qubits)
                occupation_set = _occupation_set(index)
                parity_set = _parity_set(index - 1)
                x1 = update_set
                y1 = []
                z1 = parity_set
                x2 = update_set - {index}
                y2 = {index}
                z2 = (parity_set ^ occupation_set) - {index}
                transformed_term *= _transform_ladder_operator(
                    ladder_operator, x1, y1, z1, x2, y2, z2)
            transf_op += transformed_term

        return transf_op

    def bravyi_kitaev_superfast(self):
        r"""
        Apply Bravyi-Kitaev Superfast transform.
        Implementation from https://arxiv.org/pdf/1712.00446.pdf

        Note that only hermitian operators of form

        .. math::

            C + \sum_{p, q} h_{p, q} a^\dagger_p a_q +
                \sum_{p, q, r, s} h_{p, q, r, s} a^\dagger_p a^\dagger_q a_r a_s

        where :math:`C` is a constant, be transformed.

        Returns:
            QubitOperator, qubit operator after bravyi_kitaev_superfast.
        """
        if not isinstance(self.operator, FermionOperator):
            raise TypeError(
                'This method can be only applied for FermionOperator.')
        # Get operator in normal order
        fermion_operator = normal_ordered(self.operator)

        # Get antisymmetric adjacency matrix for graph based on fermion
        # operator
        edge_matrix = _get_edge_matrix(fermion_operator)
        # Enumerate edges of the graph
        edge_enum = _enumerate_edges(edge_matrix)

        # Number of fermionic modes
        # Number of qubits
        self.n_qubits = len(edge_enum) // 2

        # Initialize identity matrix
        transf_op = QubitOperator((), fermion_operator.terms[()])
        transformed_terms = [()]

        for term in fermion_operator:
            # Check whether term is already transformed
            if term not in transformed_terms:
                at = [i for i, t in term[:len(term) // 2]]
                a = [i for i, t in term[len(term) // 2:]]
                u = set(at) | set(a)

                # Second term in pair to transform
                term_t = tuple((i, 1) for i in a) + tuple((j, 0) for j in at)

                # Check equality between numbers of creation and annihilation
                # operators in term
                if len(at) != len(a):
                    raise ValueError(
                        "Terms in hamiltonian must consist "
                        "f pairs of creation/annihilation operators")

                # Check whether fermion operator is hermitian
                if abs(fermion_operator.terms[term] -
                       fermion_operator.terms[term_t]) > 1e-8:
                    raise ValueError('Fermion operator must be hermitian.')

                # Case of a^i aj
                if len(at) == 1:
                    # Case of number operator (i=j)
                    if len(u) == 1:
                        i = u.pop()
                        transf_op += _transformed_number_operator(i,
                                                                  edge_matrix,
                                                                  edge_enum) *\
                            fermion_operator.terms[term]
                    # Case of excitation operator
                    else:
                        i, j = at[0]
                        transf_op += _transformed_excitation_operator(
                            i, j, edge_matrix,
                            edge_enum) * fermion_operator.terms[term]

                # Case of a^i a^j ak al
                elif len(at) == 2:
                    # Case of Coulomb/exchange operator (a^i ai a^j aj)
                    if len(u) == 2:
                        i, j = at[0], at[1]
                        transf_op += _transformed_exchange_operator(
                            i,
                            j,
                            edge_matrix,
                            edge_enum) *\
                            fermion_operator.terms[term] * (-1)
                        # -1 factor because of normal ordering (a^i a^j ai aj,
                        # for i>j)

                    # Case of number excitation operator (a^i a^j aj ak)
                    elif len(u) == 3:
                        i = (u - set(a)).pop()
                        j = (set(at) & set(a)).pop()
                        k = (u - set(at)).pop()
                        transf_op += _transformed_number_excitation_operator(
                            i,
                            j,
                            k,
                            edge_matrix,
                            edge_enum) *\
                            fermion_operator.terms[term] * \
                            (-1)**((i > j) ^ (j > k))
                        # -1 factor because of normal ordering

                    # Case of double excitation operator
                    elif len(u) == 4:
                        i, j, k, _ = at[0], at[1], a[0], a[1]
                        transf_op += _transformed_double_excitation_operator(
                            at[0], at[1], a[0], a[1], edge_matrix,
                            edge_enum) * fermion_operator.terms[term]

                # Adding terms in transformed pair
                transformed_terms.append(term)
                transformed_terms.append(term_t)

        return transf_op

    def ternary_tree(self):
        """
        Apply Ternary tree transform.
        Implementation from https://arxiv.org/pdf/1910.10746.pdf.

        Returns:
            QubitOperator, qubit operator after ternary_tree transformation.
        """
        h = floor(log(2 * self.n_qubits + 1, 3))
        d = self.n_qubits - (3**h - 1) // 2

        if not isinstance(self.operator, FermionOperator):
            raise TypeError(
                'This method can be only applied for FermionOperator.')
        transf_op = QubitOperator()
        for term in self.operator.terms:
            # Initialize identity matrix.
            transformed_term = QubitOperator((), self.operator.terms[term])

            # Loop through operators, transform and multiply.
            for ladder_operator in term:
                # Define lists of qubits to apply Pauli gates
                index = ladder_operator[0]

                p1 = ([2 * index // (3**l) % 3 for l in range(h, -1, -1)] if
                      2 * index < 3 * d else [(2 * index - 2 * d) // (3**l) % 3
                                              for l in range(h - 1, -1, -1)])
                x1 = []
                y1 = []
                z1 = []
                for l, tmp in enumerate(p1):
                    if tmp == 0:
                        x1 += [_get_qubit_index(p1, l)]
                    elif tmp == 1:
                        y1 += [_get_qubit_index(p1, l)]
                    else:
                        z1 += [_get_qubit_index(p1, l)]

                p2 = ([(2 * index + 1) // (3**l) % 3
                       for l in range(h, -1, -1)] if 2 * index < 3 * d else
                      [(2 * index + 1 - 2 * d) // (3**l) % 3
                       for l in range(h - 1, -1, -1)])
                x2 = []
                y2 = []
                z2 = []
                for l, tmp in enumerate(p2):
                    if tmp == 0:
                        x2 += [_get_qubit_index(p2, l)]
                    elif tmp == 1:
                        y2 += [_get_qubit_index(p2, l)]
                    else:
                        z2 += [_get_qubit_index(p2, l)]

                transformed_term *= _transform_ladder_operator(
                    ladder_operator, x1, y1, z1, x2, y2, z2)
            transf_op += transformed_term

        return transf_op

    def reversed_jordan_wigner(self):
        """
        Apply reversed Jordan-Wigner transform.

        Returns:
            FermionOperator, fermion operator after reversed_jordan_wigner transformation.
        """
        if not isinstance(self.operator, QubitOperator):
            raise TypeError(
                'This method can be only applied for QubitOperator.')
        transf_op = FermionOperator()

        # Loop through terms.
        for term in self.operator.terms:
            transformed_term = FermionOperator(())
            if term:
                working_term = QubitOperator(term)
                pauli_operator = term[-1]
                while pauli_operator is not None:

                    # Handle Pauli Z.
                    if pauli_operator[1] == 'Z':
                        transformed_pauli = FermionOperator(
                            ()) + number_operator(self.n_qubits,
                                                  pauli_operator[0], -2.)

                    # Handle Pauli X and Y.
                    else:
                        raising_term = FermionOperator(
                            ((pauli_operator[0], 1),))
                        lowering_term = FermionOperator(
                            ((pauli_operator[0], 0),))
                        if pauli_operator[1] == 'Y':
                            raising_term *= 1.j
                            lowering_term *= -1.j

                        transformed_pauli = raising_term + lowering_term

                        # Account for the phase terms.
                        for j in reversed(range(pauli_operator[0])):
                            z_term = QubitOperator(((j, 'Z'),))
                            working_term = z_term * working_term
                        term_key = list(working_term.terms)[0]
                        transformed_pauli *= working_term.terms[term_key]
                        working_term.terms[list(working_term.terms)[0]] = 1.

                    # Get next non-identity operator acting below
                    # 'working_qubit'.
                    working_qubit = pauli_operator[0] - 1
                    for working_operator in reversed(
                            list(working_term.terms)[0]):
                        if working_operator[0] <= working_qubit:
                            pauli_operator = working_operator
                            break
                        pauli_operator = None

                    # Multiply term by transformed operator.
                    transformed_term *= transformed_pauli
            # Account for overall coefficient
            transformed_term *= self.operator.terms[term]
            transf_op += transformed_term
        return transf_op


def _get_qubit_index(p, l):
    n = (3**l - 1) // 2
    for j in range(l):
        n += 3**(l - 1 - j) * p[j]
    return n


def _transform_ladder_operator(ladder_operator, x1, y1, z1, x2, y2, z2):
    r"""
    Makes transformation to qubits for :math:`a`, :math:`a^\dagger` operators:

    .. math::

        a = 1/2 X(x1) Y(y1) Z (z1)+ i/2 X(x2) Y(y2) Z(z2)

        a^\dagger = 1/2 X(x1) Y(y1) Z (z1)- i/2 X(x2) Y(y2) Z(z2)

    Args:
        ladder_operator (tuple[int, int]): the ladder operator
        n_qubits (int): the number of qubits
        x1, y1, z1, x2, y2, z2 (list[int] or set[int]):
        lists or sets of indices of qubits,
        to which corresponding Pauli gates are applied.
        These lists (sets) are defined for each transform.

    Returns:
        QubitOperator
    """
    coefficient_1 = 0.5
    coefficient_2 = -.5j if ladder_operator[1] else .5j
    transf_op_1 = QubitOperator(
        tuple((index, 'X') for index in x1) + tuple(
            (index, 'Y') for index in y1) + tuple(
                (index, 'Z') for index in z1), coefficient_1)
    transf_op_2 = QubitOperator(
        tuple((index, 'X') for index in x2) + tuple(
            (index, 'Y') for index in y2) + tuple(
                (index, 'Z') for index in z2), coefficient_2)
    return transf_op_1 + transf_op_2


def _update_set(index, n_qubits):
    """
    The bits that need to be updated upon flipping the occupancy
    of a mode. Used in Bravyi-Kitaev transform.
    """
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

    while index <= n_qubits:
        indices.add(index - 1)
        # Add least significant one to index
        # E.g. 00010100 -> 00011000
        index += index & -index
    return indices


def _occupation_set(index):
    """
    The bits whose parity stores the occupation of mode `index`.
    Used in Bravyi-Kitaev transform.
    """
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

    indices.add(index - 1)
    parent = index & (index - 1)
    index -= 1
    while index != parent:
        indices.add(index - 1)
        # Remove least significant one from index
        # E.g. 00010100 -> 00010000
        index &= index - 1
    return indices


def _parity_set(index):
    """
    The bits whose parity stores the parity of the bits 0 .. `index`.
    Used in Bravyi-Kitaev transform.
    """
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

    while index > 0:
        indices.add(index - 1)
        # Remove least significant one from index
        # E.g. 00010100 -> 00010000
        index &= index - 1
    return indices


def _get_edge_matrix(fermion_operator):
    """
    Return antisymmetric adjacency matrix (Edge matrix) for graph based on fermion operator for BKSF transform
    """
    edge_set = set()
    for term in fermion_operator:
        a = {1: [], 0: []}
        for ladder_operator in term:
            if ladder_operator[0] in a[ladder_operator[1] ^ 1]:
                a[ladder_operator[1] ^ 1].remove(ladder_operator[0])
            else:
                a[ladder_operator[1]].append(ladder_operator[0])

        if len(a[1]) == 2:
            edge_set.add(tuple(a[1]))
            edge_set.add(tuple(a[0]))
        elif len(a[1]) == 1:
            a = [*a[1], *a[0]] if a[1][0] > a[0][0] else [*a[0], *a[1]]
            edge_set.add(tuple(a))

    d = count_qubits(fermion_operator)
    edge_matrix = np.zeros((d, d))

    for i, j in edge_set:
        edge_matrix[i, j] = -1
        edge_matrix[j, i] = 1

    return edge_matrix


def _enumerate_edges(edge_matrix):
    """
    Return dictionary of edges of the graph and its coresponing number based
    on Edge matrix
    """
    d = len(edge_matrix)
    edge_enum = {}
    n = 0
    for i in range(d):
        for j in range(i + 1, d):
            if edge_matrix[i, j] > 0:
                edge_enum[(i, j)] = n
                edge_enum[(j, i)] = n
                n = n + 1

    return edge_enum


def _get_b(i, edge_matrix, edge_enum):
    """Return b_i qubit operator based on Edge matrix"""
    long_string = ''
    for j in range(len(edge_matrix[i])):
        if edge_matrix[i, j] != 0:
            long_string += 'Z' + str(edge_enum[(i, j)]) + ' '
    return QubitOperator(long_string)


def _get_a(i, j, edge_matrix, edge_enum):
    """Return a_ij qubit operator based on Edge matrix"""
    long_string = ''
    long_string = 'X' + str(edge_enum[(i, j)]) + ' '

    for l in range(0, j):
        if edge_matrix[l, i] != 0:
            long_string += 'Z' + str(edge_enum[(l, i)]) + ' '
    for s in range(0, i):
        if edge_matrix[s, j] != 0:
            long_string += 'Z' + str(edge_enum[(s, j)]) + ' '

    return QubitOperator(long_string, edge_matrix[i, j])


def _transformed_number_operator(i, edge_matrix, edge_enum):
    """Return qubit operator based on Edge matrix for a^i ai term"""
    return (QubitOperator(()) - _get_b(i, edge_matrix, edge_enum)) / 2


def _transformed_excitation_operator(i, j, edge_matrix, edge_enum):
    """Return qubit operator based on Edge matrix for a^i aj + a^j ai term"""
    a_ij = _get_a(i, j, edge_matrix, edge_enum)
    return (a_ij * _get_b(j, edge_matrix, edge_enum) +
            _get_b(i, edge_matrix, edge_enum) * a_ij) * (-1j / 2)


def _transformed_exchange_operator(i, j, edge_matrix, edge_enum):
    """Return qubit operator based on Edge matrix for a^i ai a^j aj"""
    return (QubitOperator(()) - _get_b(i, edge_matrix, edge_enum)) *\
        (QubitOperator(()) - _get_b(j, edge_matrix, edge_enum)) / 4


def _transformed_number_excitation_operator(i, j, k, edge_matrix, edge_enum):
    """Return qubit operator based on Edge matrix for a^i a^j aj ak + a^k a^j aj ai"""
    a_ik = _get_a(i, k, edge_matrix, edge_enum)
    return (a_ik * _get_b(k, edge_matrix, edge_enum) +
            _get_b(i, edge_matrix, edge_enum) * a_ik) * (QubitOperator(
                ()) - _get_b(j, edge_matrix, edge_enum)) * (-1j / 4)


def _transformed_double_excitation_operator(i, j, k, l, edge_matrix,
                                            edge_enum):
    """Return qubit operator based on Edge matrix for a^i a^j ak al"""
    b_i = _get_b(i, edge_matrix, edge_enum)
    b_j = _get_b(j, edge_matrix, edge_enum)
    b_k = _get_b(k, edge_matrix, edge_enum)
    b_l = _get_b(l, edge_matrix, edge_enum)

    return _get_a(i, j, edge_matrix, edge_enum) * _get_a(k, l, edge_matrix,
                                                         edge_enum) *\
        (-QubitOperator(()) - (b_i * b_j + b_k * b_l) + b_i * b_k +
         b_i * b_l + b_j * b_k + b_j * b_l + b_i * b_j * b_k * b_l) / 8
