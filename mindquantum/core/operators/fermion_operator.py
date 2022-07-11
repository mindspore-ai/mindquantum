# Portions Copyright 2021 Huawei Technologies Co., Ltd
# Portions Copyright 2017 The OpenFermion Developers.
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

# pylint: disable=duplicate-code

"""This module is generated the Fermion Operator."""

import json
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix, kron

from mindquantum.utils.type_value_check import _check_input_type, _check_int_type

from ..parameterresolver import ParameterResolver
from ._base_operator import _Operator


@lru_cache()
def _n_sz(n):
    if n == 0:
        return csr_matrix(np.array([1]), dtype=np.complex128)
    tmp = [csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128)) for _ in range(n)]
    for i in tmp[1:]:
        tmp[0] = kron(tmp[0], i)
    return tmp[0]


@lru_cache()
def _n_identity(n):
    """N_identity."""
    if n == 0:
        return csr_matrix(np.array([1]), dtype=np.complex128)
    tmp = [csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.complex128)) for _ in range(n)]
    for i in tmp[1:]:
        tmp[0] = kron(tmp[0], i)
    return tmp[0]


@lru_cache()
def _single_fermion_word(idx, dag, n_qubits):
    """Single_fermion_word."""
    matrix = csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.complex128))
    if dag:
        matrix = csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.complex128))
    return kron(_n_identity(n_qubits - 1 - idx), kron(matrix, _n_sz(idx)))


@lru_cache()
def _two_fermion_word(idx1, dag1, idx2, dag2, n_qubits):
    """Two_fermion_word."""
    return _single_fermion_word(idx1, dag1, n_qubits) * _single_fermion_word(idx2, dag2, n_qubits)


def _check_valid_fermion_operator_term(fo_term):
    """Check valid fermion operator term."""
    if fo_term is not None and fo_term != '':
        if not isinstance(fo_term, (str, tuple)):
            raise ValueError(f'Fermion operator requires a string or a tuple, but get {type(fo_term)}')
        if isinstance(fo_term, str):
            terms = fo_term.split(' ')
            for term in terms:
                if (term.endswith('^') and not term[:-1].isdigit()) or (not term.endswith('^') and not term.isdigit()):
                    if term:
                        raise ValueError(f'Invalid fermion operator term {term}')
        if isinstance(fo_term, tuple):
            for term in fo_term:
                if (
                    len(term) != 2
                    or not isinstance(term[0], int)
                    or not isinstance(term[1], int)
                    or term[0] < 0
                    or term[1] not in [0, 1]
                ):
                    raise ValueError(f'Invalid fermion operator term {term}')


class FermionOperator(_Operator):
    r"""
    Definition of a Fermion Operator.

    The Fermion Operator such as FermionOperator(' 4^ 3 9 3^ ') are used to represent :math:`a_4^\dagger a_3 a_9
    a_3^\dagger`.


    These are the Basic Operators to describe a fermionic system, such as a Molecular system.
    The FermionOperator are follows the anti-commutation relationship.

    Args:
        terms (str): The input term of fermion operator. Default: None.
        coefficient (Union[numbers.Number, str, ParameterResolver]): The coefficient for the corresponding single
            operators Default: 1.0.

    Examples:
        >>> from mindquantum.core.operators import FermionOperator
        >>> a_p_dagger = FermionOperator('1^')
        >>> a_p_dagger
        1.0 [1^]
        >>> a_q = FermionOperator('0')
        >>> a_q
        1.0 [0]
        >>> zero = FermionOperator()
        >>> zero
        0
        >>> identity= FermionOperator('')
        >>> identity
        1.0 []
        >>> para_op = FermionOperator('0 1^', 'x')
        >>> para_op
        x [0 1^]
        >>> para_dt = {'x':2}
        >>> op = para_op.subs(para_dt)
        >>> op
        2 [0 1^]
    """

    __hash__ = None

    def __init__(self, term=None, coefficient=1.0):
        """Initialize a FermionOperator object."""
        super().__init__(term, coefficient)
        _check_valid_fermion_operator_term(term)
        self.operators = {1: '^', 0: '', '^': '^', '': ''}
        self.gates_number = 0
        self.qubit_type = False

        if term is not None:
            if term == '':
                term = self._parse_term(())
            else:
                term = self._parse_term(term)
            self.terms[term] = self.coefficient

    def _simplify(self, terms, coefficient=1.0):
        """Simplify a term."""
        return coefficient, tuple(terms)

    def _parse_string(self, terms_string):
        """
        Parse a term given as a string type.

        e.g. For FermionOperator:
                 4^ 3  -> ((4, 1),(3, 0))
        Note here the '1' and '0' in the second col represents creation and annihilaiton operator respectively

        Returns:
            tuple, return a tuple list, such as ((4, 1),(3, 0))

        Raises:
            '1.5 4^ 3' is not the proper format and
            could raise TypeError.
        """

        def map_operator_to_integer_rep(operator):
            """Map operator to integer."""
            return 1 if operator == '^' else 0

        terms = terms_string.split()
        terms_to_tuple = []
        for sub_term in terms:
            index = int(sub_term[0])
            operator = sub_term[1:]
            # Handle such cases: 10^, 100^, ...
            if len(sub_term) >= 2:
                if '^' in sub_term:
                    operator = '^'
                    index = int(sub_term[: sub_term.index(operator)])
                else:
                    operator = ''
                    index = int(sub_term)

            if operator not in self.operators:
                raise ValueError(
                    f'Invalid type of operator {operator}.'
                    f'The Fermion operator should be one of this {self.operators}'
                )
            if index < 0:
                raise ValueError(f"Invalid index {self.operators}.The qubit index should be non negative integer")
            terms_to_tuple.append((index, map_operator_to_integer_rep(operator)))
            # check the commutate terms with same index in the list and
            # replace it with the corresponding commutation relationship
        return tuple(terms_to_tuple)

    def to_openfermion(self):
        """Convert fermion operator to openfermion format."""
        # pylint: disable=import-outside-toplevel
        from openfermion import FermionOperator as OFFermionOperator

        terms = {}
        for k, v in self.terms.items():
            if not v.is_const():
                raise ValueError("Cannot convert parameteized fermion operator to openfermion format")
            terms[k] = v.const
        fermion_operator = OFFermionOperator()
        fermion_operator.terms = terms
        return fermion_operator

    @staticmethod
    def from_openfermion(of_ops):
        """
        Convert openfermion fermion operator to mindquantum format.

        Args:
            of_ops (openfermion.FermionOperator): fermion operator from openfermion.

        Returns:
            FermionOperator, fermion operator from mindquantum.
        """
        # pylint: disable=import-outside-toplevel
        from openfermion import FermionOperator as OFFermionOperator

        _check_input_type('of_ops', OFFermionOperator, of_ops)
        fermion_operator = FermionOperator()
        for k, v in of_ops.terms.items():
            fermion_operator.terms[k] = ParameterResolver(v)
        return fermion_operator

    def __str__(self):  # pylint: disable=too-many-branches
        """Return an easy-to-read string representation of the FermionOperator."""
        if not self.terms:
            return '0'
        string_rep = ''
        term_cnt = 0
        for term, coeff in sorted(self.terms.items()):
            term_cnt += 1
            if isinstance(coeff, ParameterResolver):
                tmp_string = f'{coeff.expression()} ['  # begin of the '['
            else:
                tmp_string = f'{coeff} ['  # begin of the '['
            # deal with this situation (1,'X') or [1, 'X']
            if term == ():
                if self.size == 1:
                    tmp_string.join(' ]')
                else:
                    pass

            elif isinstance(term[0], int):
                index, operator = term
                if operator in self.operators:
                    tmp_string += f'{index}{self.operators[operator]} '
            else:
                for sub_term in term:
                    index, operator = sub_term
                    # check validity, if checked before,
                    # then we can take away this step
                    if operator in self.operators:
                        tmp_string += f'{index}{self.operators[operator]} '

            if term_cnt < len(self.terms):
                string_rep += f'{tmp_string.strip()}] +\n'  # end of the ']'
            else:
                string_rep += f'{tmp_string.strip()}] '  # end of the ']'

        return string_rep

    def __repr__(self):
        """Return a string representation of the object."""
        return str(self)

    def matrix(self, n_qubits=None):  # pylint: disable=too-many-branches
        """
        Convert this fermion operator to csr_matrix under jordan_wigner mapping.

        Args:
            n_qubits (int): The total qubit of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        from .utils import (  # pylint: disable=import-outside-toplevel,cyclic-import
            count_qubits,
        )

        if not self.terms:
            raise ValueError("Cannot convert empty fermion operator to matrix")
        n_qubits_local = count_qubits(self)
        if n_qubits_local == 0 and n_qubits is None:
            raise ValueError("You should specific n_qubits for converting a identity fermion operator.")
        if n_qubits is None:
            n_qubits = n_qubits_local
        _check_int_type("n_qubits", n_qubits)
        if n_qubits < n_qubits_local:
            raise ValueError(
                f"Given n_qubits {n_qubits} is small than qubit of fermion operator, which is {n_qubits_local}."
            )
        out = 0
        for term, coeff in self.terms.items():
            if not coeff.is_const():
                raise RuntimeError("Cannot convert a parameterized fermion operator to matrix.")
            coeff = coeff.const
            if not term:
                out += csr_matrix(np.identity(2**n_qubits, dtype=np.complex128)) * coeff
            else:
                tmp = 1
                group = [[]]
                for idx, dag in term:
                    if len(group[-1]) < 4:
                        group[-1].append(idx)
                        group[-1].append(dag)
                    if len(group[-1]) == 4:
                        group.append([])
                for gate in group:
                    if gate:
                        if len(gate) == 4:
                            tmp *= _two_fermion_word(gate[0], gate[1], gate[2], gate[3], n_qubits)
                        else:
                            tmp *= _single_fermion_word(gate[0], gate[1], n_qubits)
                out += tmp * coeff
        return out

    @property
    def imag(self):
        """
        Convert the coefficient to its imag part.

        Returns:
            FermionOperator, the imag part of this fermion operator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.imag.compress()
            2.0 [0]
        """
        out = FermionOperator()

        for term, coeff in self.terms.items():
            out.terms[term] = coeff.imag
        return out

    @property
    def real(self):
        """
        Convert the coefficient to its real part.

        Returns:
            FermionOperator, the real part of this fermion operator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.real.compress()
            1.0 [0] +
            a [0^]
        """
        out = FermionOperator()

        for term, coeff in self.terms.items():
            out.terms[term] = coeff.real
        return out

    def normal_ordered(self):
        """
        Return the normal ordered form of the Fermion Operator.

        Returns:
            FermionOperator, the normal ordered FermionOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> origin = FermionOperator('0 1^')
            >>> origin
            1.0 [0 1^]
            >>> origin.normal_ordered()
            -1.0 [1^ 0]

        """
        ordered_op = self.__class__()
        for term, coeff in self.terms.items():
            ordered_op += _normal_ordered_term(term, coeff)
        return ordered_op

    def dumps(self, indent=4):
        r"""
        Dump FermionOperator into JSON(JavaScript Object Notation).

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            JSON (str), the JSON strings of FermionOperator

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> len(f.dumps())
            443
        """
        if indent is not None:
            _check_int_type('indent', indent)
        dic = {}
        for term, coeff in self.terms.items():
            string = _fermion_tuple_to_string(term)
            dic[string] = coeff.dumps(indent)
        return json.dumps(dic, indent=indent)

    @staticmethod
    def loads(strs):
        """
        Load JSON(JavaScript Object Notation) into FermionOperator.

        Args:
            strs (str): The dumped fermion operator string.

        Returns:
            FermionOperator, the FermionOperator load from strings

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> obj = FermionOperator.loads(f.dumps())
            >>> obj == f
            True
        """
        _check_input_type('strs', str, strs)
        dic = json.loads(strs)
        f_op = FermionOperator()
        for k, v in dic.items():
            f_op += FermionOperator(k, ParameterResolver.loads(v))
        return f_op

    def split(self):
        """
        Split the coefficient and the operator.

        Returns:
            List[List[ParameterResolver, FermionOperator]], the split result.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> a = FermionOperator('0', 'a') + FermionOperator('1^', 1.2)
            >>> list(a.split())
            [[{'a': 1}, const: 0, 1 [0] ], [{}, const: 1.2, 1 [1^] ]]
        """
        for term, coeff in self.terms.items():
            yield [coeff, FermionOperator(term)]


def _normal_ordered_term(term, coefficient):
    r"""
    Return the normal ordered term of the FermionOperator with high index and creation operator in front.

    eg. :math:`a_3\dagger a_2\dagger a_1 a_0`

    """
    term = list(term)
    ordered_term = FermionOperator()
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            left_sub_term = term[j - 1]
            right_sub_term = term[j]
            # Swap operators if left operator is a and right operator is
            # a^\dagger
            if not left_sub_term[1] and right_sub_term[1]:
                term[j], term[j - 1] = left_sub_term, right_sub_term
                coefficient = -1 * coefficient
                # If indice are same, employ the anti-commutation relationship
                # And generate the new term
                if left_sub_term[0] == right_sub_term[0]:
                    new_term = term[: (j - 1)] + term[(j + 1) :]  # noqa: E203
                    ordered_term += _normal_ordered_term(new_term, -1 * coefficient)
            elif left_sub_term[1] == right_sub_term[1]:
                # If indice are same,evaluate it to zero.
                if left_sub_term[0] == right_sub_term[0]:
                    return ordered_term
                # Swap them if same operator but lower index on left
                if left_sub_term[0] < right_sub_term[0]:
                    term[j], term[j - 1] = left_sub_term, right_sub_term
                    coefficient = -1 * coefficient

    # Add the term and return.
    ordered_term += FermionOperator(_fermion_tuple_to_string(tuple(term)), coefficient)
    return ordered_term


def _fermion_tuple_to_string(term):
    string = []
    for i in term:
        if i[1] == 1:
            string.append(f'{i[0]}^')
        else:
            string.append(str(i[0]))
    return ' '.join(string)
