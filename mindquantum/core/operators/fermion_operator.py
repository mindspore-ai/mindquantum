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

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, kron

from mindquantum.core.operators._base_operator import EQ_TOLERANCE
from mindquantum.experimental._mindquantum_cxx.ops import (
    FermionOperatorPR as FermionOperator_,
)
from mindquantum.experimental.utils import TermValueCpp, TermValueStr
from mindquantum.utils.type_value_check import _check_input_type

from ..parameterresolver import ParameterResolver


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


class FermionOperator(FermionOperator_):
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

    def __init__(self, term=None, coeff=1.0):
        """Initialize a FermionOperator object."""
        if isinstance(term, FermionOperator_):
            FermionOperator_.__init__(self, term)
        else:
            if term is None:
                FermionOperator_.__init__(self)
            if isinstance(term, dict):
                FermionOperator_.__init__(self, term)
            else:
                if not isinstance(coeff, ParameterResolver):
                    coeff = ParameterResolver(coeff)
                FermionOperator_.__init__(self, term, coeff)

    def __deepcopy__(self, memodict) -> "FermionOperator":
        """Deep copy this FermionOperator."""
        return FermionOperator(self)

    def __str__(self) -> str:
        """Return string expression of FermionOperator."""
        terms = self.terms
        new_str = ''
        for idx, (term, coeff) in enumerate(terms.items()):
            term = [f"{i}{'^' if j else ''}" for i, j in term]
            end = ' +\n'
            if idx == len(terms) - 1:
                end = ' '
            new_str += f"{coeff.expression()} [{' '.join(term)}]{end}"
        return new_str if new_str else "0"

    def __repr__(self) -> str:
        """Return string expression of FermionOperator."""
        return self.__str__()

    def __iter__(self) -> "FermionOperator":
        """Iterate every single term."""
        for term, coeff in self.terms.items():
            yield FermionOperator(term, coeff)

    def __len__(self) -> int:
        """Return the size of term."""
        return self.size

    def __neg__(self) -> "FermionOperator":
        """Return negative FermionOperator."""
        return FermionOperator(FermionOperator_.__neg__(self))

    def __add__(self, other) -> "FermionOperator":
        """Add a number or a FermionOperator."""
        return FermionOperator(FermionOperator_.__add__(self, other))

    def __iadd__(self, other) -> "FermionOperator":
        """Inplace add a number or a FermionOperator."""
        FermionOperator_.__iadd__(self, other)
        return self

    def __radd__(self, other) -> "FermionOperator":
        """Right add a number or a FermionOperator."""
        return FermionOperator(FermionOperator_.__add__(self, other))

    def __sub__(self, other) -> "FermionOperator":
        """Subtract a number or a FermionOperator."""
        return FermionOperator(FermionOperator_.__sub__(self, other))

    def __isub__(self, other) -> "FermionOperator":
        """Inplace subtrace a number or a FermionOperator."""
        FermionOperator_.__isub__(self, other)
        return self

    def __rsub__(self, other) -> "FermionOperator":
        """Subtrace a number or a FermionOperator with this FermionOperator."""
        return other + (-self)

    def __mul__(self, other) -> "FermionOperator":
        """Multiple a number or a FermionOperator."""
        if isinstance(other, str):
            other = ParameterResolver(other)
        return FermionOperator(FermionOperator_.__mul__(self, other))

    def __imul__(self, other) -> "FermionOperator":
        """Inplace multiple a number or a FermionOperator."""
        if isinstance(other, str):
            other = ParameterResolver(other)
        FermionOperator_.__imul__(self, other)
        return self

    def __rmul__(self, other) -> "FermionOperator":
        """Right multiple a number or a FermionOperator."""
        if isinstance(other, str):
            other = ParameterResolver(other)
        return FermionOperator(FermionOperator_.__mul__(self, other))

    def __truediv__(self, other) -> "FermionOperator":
        """Divide a number."""
        return FermionOperator(FermionOperator_.__truediv__(self, other))

    def __itruediv__(self, other) -> "FermionOperator":
        """Divide a number."""
        FermionOperator_.__itruediv__(self, other)
        return self

    def __power__(self, exponent: int) -> "FermionOperator":
        """Exponential of FermionOperator."""
        return FermionOperator(FermionOperator_.__power__(self, exponent))

    def __eq__(self, other) -> bool:
        """Check whether two FermionOperator equal."""
        return FermionOperator_.__eq__(self, other)

    def __ne__(self, other) -> bool:
        """Check whether two FermionOperator not equal."""
        return not FermionOperator_.__eq__(self, other)

    @property
    def imag(self) -> "FermionOperator":
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
        return FermionOperator(FermionOperator_.imag(self))

    @property
    def real(self) -> "FermionOperator":
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
        return FermionOperator(FermionOperator_.real(self))

    @property
    def terms(self) -> Dict[Tuple[Tuple[int]], ParameterResolver]:
        """Get the term of FermionOperator."""
        return {tuple(i): ParameterResolver(j) for i, j in FermionOperator_.terms(self)}

    def compress(self, abs_tol=EQ_TOLERANCE) -> "FermionOperator":
        """
        Eliminate the very small terms that close to zero.

        Removes small imaginary and real parts.

        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0

        Returns:
            the compressed operator

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> ham_compress = FermionOperator('0^ 1', 0.5) + FermionOperator('2^ 3', 1e-7)
            >>> ham_compress
            1/2 [0^ 1] +
            1/10000000 [2^ 3]
            >>> ham_compress.compress(1e-6)
            1/2 [0^ 1]
            >>> ham_para_compress =  FermionOperator('0^ 1', 0.5) + FermionOperator('2^ 3', 'X')
            >>> ham_para_compress
            1/2 [0^ 1] +
            X [2^ 3]
            >>> ham_para_compress.compress(1e-7)
            1/2 [0^ 1] +
            X [2^ 3]
        """
        return FermionOperator(FermionOperator_.compress(self, abs_tol))

    @property
    def constant(self) -> "FermionOperator":
        """Return the value of the constant term."""
        return FermionOperator_.constant(self)

    @constant.setter
    def constant(self, coeff):
        """Set the coefficient of the Identity term."""
        if not isinstance(coeff, ParameterResolver):
            coeff = ParameterResolver(coeff)
        FermionOperator_.constant(self, coeff)

    def count_qubits(self) -> int:
        """
        Calculate the number of qubits on which operator acts before removing the unused qubit.

        Returns:
            int, the qubits number before remove unused qubit.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> a = FermionOperator("0^ 3")
            >>> a.count_qubits()
            4
        """
        return FermionOperator_.count_qubits(self)

    def dumps(self, indent: int = 4) -> str:
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
            922
        """
        return FermionOperator_.dumps(self, indent)

    @staticmethod
    def loads(strs: str) -> "FermionOperator":
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
        return FermionOperator(FermionOperator_.loads(strs))

    def hermitian(self) -> "FermionOperator":
        """
        Get the hermitian of FermionOperator.

        Returns:
            The hermitian of FermionOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> a = FermionOperator("0^ 1", {"a": 1 + 2j})
            >>> a.hermitian()
            (1-2j)*a [1^ 0]
        """
        return FermionOperator(FermionOperator_.hermitian(self))

    def matrix(self, n_qubits: int = None):
        """
        Convert this fermion operator to csr_matrix under jordan_wigner mapping.

        Args:
            n_qubits (int): The total qubit of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        if n_qubits is None:
            n_qubits = self.count_qubits()
        return FermionOperator(FermionOperator_.matrix(self, n_qubits))

    def normal_ordered(self) -> "FermionOperator":
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
        return FermionOperator(FermionOperator_.normal_ordered(self))

    def subs(self, params_value: ParameterResolver) -> "FermionOperator":
        """Replace the symbolical representation with the corresponding value."""
        if not isinstance(params_value, ParameterResolver):
            params_value = ParameterResolver(params_value)
        return FermionOperator(FermionOperator_.subs(self, params_value))

    @property
    def is_singlet(self) -> bool:
        """
        To verify whether this operator has only one term.

        Returns:
            bool, whether this operator has only one term.
        """
        return FermionOperator_.is_singlet(self)

    def singlet(self) -> List["FermionOperator"]:
        """
        Split the single string operator into every word.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Returns:
            List[FermionOperator]: The split word of the string.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> ops = FermionOperator("1^ 2", 1)
            >>> print(ops.singlet())
            [1 [1^] , 1 [2] ]
        """
        return [FermionOperator(i) for i in FermionOperator_.singlet(self)]

    def singlet_coeff(self) -> ParameterResolver:
        """
        Get the coefficient of this operator, if the operator has only one term.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Returns:
            ParameterResolver: the coefficient of this single string operator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> ops = FermionOperator("1^ 2", "a")
            >>> print(ops.singlet_coeff())
            {'a': (1,0)}, const: (0,0)
        """
        return ParameterResolver(FermionOperator_.singlet_coeff(self))

    @property
    def size(self):
        """Return the size of the FermionOperator terms."""
        return FermionOperator_.size(self)

    # TODO(xusheng): Finish type hint.
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
        for i, j in FermionOperator_.split(self):
            yield [ParameterResolver(i), FermionOperator(j)]

    def to_openfermion(self):
        """Convert fermion operator to openfermion format."""
        # pylint: disable=import-outside-toplevel
        from openfermion import FermionOperator as OFFermionOperator

        terms = {}
        for term, pr in self.terms.items():
            if not pr.is_const:
                raise ValueError("Cannot convert parameteized fermion operator to openfermion format")
            terms[term] = pr.const
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
        terms = {}
        for k, v in of_ops.terms.items():
            terms[tuple((i, TermValueCpp[TermValueStr[j]]) for i, j in k)] = ParameterResolver(v)
        return FermionOperator(terms)


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
