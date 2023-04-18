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
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# pylint: disable=duplicate-code
"""This module implements qubit-excitation operators."""

from mindquantum.core.parameterresolver import ParameterResolver

from ._base_operator import _Operator
from ._term_value import TermValue
from .fermion_operator import FermionOperator
from .qubit_operator import QubitOperator


def _check_valid_qubit_excitation_operator_term(qeo_term):
    """Check valid qubit excitation operator term."""
    if qeo_term is not None and qeo_term != '':
        if not isinstance(qeo_term, (str, tuple)):
            raise ValueError(f'Qubit excitation operator requires a string or a tuple, but get {type(qeo_term)}')
        if isinstance(qeo_term, str):
            terms = qeo_term.split(' ')
            for term in terms:
                if (term.endswith('^') and not term[:-1].isdigit()) or (not term.endswith('^') and not term.isdigit()):
                    if term:
                        raise ValueError(f'Invalid qubit excitation operator term {term}')
        if isinstance(qeo_term, tuple):
            for term in qeo_term:
                if (
                    len(term) != 2
                    or not isinstance(term[0], int)
                    or term[0] < 0
                    or term[1] not in (TermValue[0], TermValue[1], 0, 1)
                ):
                    raise ValueError(f'Invalid qubit excitation operator term {term}')


class QubitExcitationOperator(_Operator):
    r"""
    QubitExcitationOperator class.

    The Qubit Excitation Operator is defined as:
    :math:`Q^{\dagger}_{n} = \frac{1}{2} (X_{n} - iY_{n})` and
    :math:`Q_{n} = \frac{1}{2} (X_{n} + iY_{n})`. Compared with Fermion
    excitation operators, Qubit excitation operators are some kind of
    "localized", i.e., the Fermion excitation operator
    :math:`a^{\dagger}_{7} a_{0}` involves qubit ranging from 0 to 7 under JW
    transformation, while Qubit excitation :math:`Q^{\dagger}_{7} Q_{0}` will
    only affect the 0th and 7th qubits. In addition, double excitations
    described using Qubit excitation operators use much less CNOTs than the
    corresponding Fermion excitation operators.

    Args:
        terms (Union[str, tuple]): The input term of qubit excitation operator. Default: ``None``.
        coefficient (Union[numbers.Number, str, ParameterResolver]): The
            coefficient for the corresponding single operators Default: ``1.0``.

    Examples:
        >>> from mindquantum.algorithm.nisq import Transform
        >>> from mindquantum.core.operators import QubitExcitationOperator
        >>> op = QubitExcitationOperator(((4, 1), (1, 0), (0, 0)), 2.5)
        >>> op
        5/2 [Q4^ Q1 Q0]
        >>> op.fermion_operator
        5/2 [4^ 1 0]
        >>> op.to_qubit_operator()
        5/16 [X0 X1 X4] +
        (-0.3125j) [X0 X1 Y4] +
        (5/16j) [X0 Y1 X4] +
        5/16 [X0 Y1 Y4] +
        (5/16j) [Y0 X1 X4] +
        5/16 [Y0 X1 Y4] +
        -0.3125 [Y0 Y1 X4] +
        (5/16j) [Y0 Y1 Y4]
        >>> Transform(op.fermion_operator).jordan_wigner()
        5/16 [X0 X1 Z2 Z3 X4] +
        (-0.3125j) [X0 X1 Z2 Z3 Y4] +
        (5/16j) [X0 Y1 Z2 Z3 X4] +
        5/16 [X0 Y1 Z2 Z3 Y4] +
        (5/16j) [Y0 X1 Z2 Z3 X4] +
        5/16 [Y0 X1 Z2 Z3 Y4] +
        -0.3125 [Y0 Y1 Z2 Z3 X4] +
        (5/16j) [Y0 Y1 Z2 Z3 Y4]
    """

    __hash__ = None

    def __init__(self, term=None, coefficient=1.0):
        """Initialize a QubitExcitationOperator object."""
        super().__init__(term, coefficient)
        _check_valid_qubit_excitation_operator_term(term)
        self.operators = {
            1: TermValue['adg'],
            0: TermValue['a'],
            '^': TermValue['adg'],
            '': TermValue['a'],
            TermValue['a']: TermValue['a'],
            TermValue['adg']: TermValue['adg'],
        }
        self.gates_number = 0
        self.qubit_type = False

        if term is not None:
            if term == '':
                term = self._parse_term(())
            else:
                term = self._parse_term(term)
            self.terms[term] = self.coefficient

        self.fermion_operator = FermionOperator(term, coefficient)

    def to_qubit_operator(self):
        r"""
        Convert the Qubit excitation operator to the equivalent Qubit operator.

        Returns:
            QubitOperator, The corresponding QubitOperator
            according to the definition of Qubit excitation operators.

        Examples:
            >>> from mindquantum.core.operators import QubitExcitationOperator
            >>> op = QubitExcitationOperator("7^ 1")
            >>> op.to_qubit_operator()
            1/4 [X1 X7] +
            (-1/4j) [X1 Y7] +
            (1/4j) [Y1 X7] +
            1/4 [Y1 Y7]
        """
        qubit_operator = QubitOperator()
        for term_i, coeff_i in self.terms.items():
            qubit_operator_i = QubitOperator((), 1)
            for (idx, excite) in term_i:
                qubit_op_ = None
                if excite == TermValue[0]:
                    qubit_op_ = QubitOperator(((idx, TermValue["X"]),), 1) + QubitOperator(((idx, TermValue["Y"]),), 1j)
                else:
                    qubit_op_ = QubitOperator(((idx, TermValue["X"]),), 1) - QubitOperator(((idx, TermValue["Y"]),), 1j)
                qubit_op_ *= 0.5
                qubit_operator_i *= qubit_op_
            qubit_operator_i *= coeff_i
            qubit_operator += qubit_operator_i
        return qubit_operator

    def hermitian(self):
        """Return Hermitian conjugate of QubitExcitationOperator."""
        conjugate_operator = QubitExcitationOperator()
        for term, coefficient in self.terms.items():
            # reverse the order and change the action from 0(1) to 1(0)
            conjugate_term = tuple(
                (index, TermValue.adg if op == TermValue.a else TermValue.a) for (index, op) in reversed(term)
            )
            conjugate_operator.terms[conjugate_term] = coefficient.conjugate()
        return conjugate_operator

    def _simplify(self, terms, coefficient=1.0):
        """Simplify a term."""
        return coefficient, tuple(terms)

    def _parse_string(self, terms_string):
        r"""
        Parse a term given as a string type.

        e.g. For QubitExcitationOperator: 4^ 3  -> ((4, 1),(3, 0))
        Note here the '1' and '0' in the second col represents creation and annihilation operator respectively

        Returns:
            tuple, return a tuple list, such as ((4, 1),(3, 0))

        Raises:
            '1.5 4^ 3' is not the proper format and
            could raise TypeError.
        """

        def map_operator_to_integer_rep(operator):
            """Operator to integer conversion function."""
            return TermValue.adg if operator == '^' else TermValue.a

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
                    f'The Qubit excitation operator should be one of this {self.operators}'
                )
            if index < 0:
                raise ValueError(f"Invalid index {self.operators}.The qubit index should be non negative integer")
            terms_to_tuple.append((index, map_operator_to_integer_rep(operator)))
            # check the commutate terms with same index in the list and
            # replace it with the corresponding commutation relationship
        return tuple(terms_to_tuple)

    def __str__(self):  # pylint: disable=too-many-branches
        """Return an easy-to-read string representation of the QubitExcitationOperator."""
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
                    tmp_string += f"Q{index}{'^' if operator == self.operators['^'] else ''} "
            else:
                for sub_term in term:
                    index, operator = sub_term
                    # check validity, if checked before,
                    # then we can take away this step
                    if operator in self.operators:
                        tmp_string += f"Q{index}{'^' if operator == self.operators['^'] else ''} "

            if term_cnt < len(self.terms):
                string_rep += f'{tmp_string.strip()}] +\n'  # end of the ']'
            else:
                string_rep += f'{tmp_string.strip()}] '  # end of the ']'

        return string_rep

    def __repr__(self):
        """Return a string representation of the object."""
        return str(self)

    @property
    def imag(self):
        """
        Convert the coefficient to its imag part.

        Returns:
            QubitExcitationOperator, the image part of this qubit excitation operator.

        Examples:
            >>> from mindquantum.core.operators import QubitExcitationOperator
            >>> f = QubitExcitationOperator(((1, 0),), 1 + 2j)
            >>> f += QubitExcitationOperator(((1, 1),), 'a')
            >>> f.imag.compress()
            2 [Q1]
        """
        out = QubitExcitationOperator()

        for k, v in self.terms.items():
            out.terms[k] = v.imag
        return out

    @property
    def real(self):
        """
        Convert the coefficient to its real part.

        Returns:
            QubitExcitationOperator, the real part of this qubit excitation operator.

        Examples:
            >>> from mindquantum.core.operators import QubitExcitationOperator
            >>> f = QubitExcitationOperator(((1, 0),), 1 + 2j)
            >>> f += QubitExcitationOperator(((1, 1),), 'a')
            >>> f.real.compress()
            1 [Q1] +
            a [Q1^]
        """
        out = QubitExcitationOperator()

        for k, v in self.terms.items():
            out.terms[k] = v.real
        return out

    def normal_ordered(self):
        r"""
        Return the normal ordered form of the Qubit excitation operator.

        Note:
            Unlike Fermion excitation operators, Qubit excitation operators
            will not multiply -1 when the order is swapped.

        Returns:
            QubitExcitationOperator, the normal ordered operator.

        Examples:
            >>> from mindquantum.core.operators import QubitExcitationOperator
            >>> op = QubitExcitationOperator("7 1^")
            >>> op
            1 [Q7 Q1^]
            >>> op.normal_ordered()
            1 [Q1^ Q7]
        """
        ordered_op = self.__class__()
        for term, coeff in self.terms.items():
            ordered_op += _normal_ordered_term(term, coeff)
        return ordered_op


def _normal_ordered_term(term, coefficient):
    r"""
    Return the normal ordered term of the QubitExcitationOperator with high index and creation operator in front.

    eg. :math:`Q_{3}^{\dagger} Q_{2}^{\dagger} Q_{1} Q_{0}`

    """
    term = list(term)
    ordered_term = QubitExcitationOperator()
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            left_sub_term = term[j - 1]
            right_sub_term = term[j]
            # Swap operators if left operator is a and right operator is
            # Q^{\dagger}
            if left_sub_term[1] == TermValue.a and right_sub_term[1] == TermValue.adg:
                term[j], term[j - 1] = left_sub_term, right_sub_term
                # If indices are same, employ the commutation relationship
                # And generate the new term
                if left_sub_term[0] == right_sub_term[0]:
                    new_term = term[: (j - 1)] + term[(j + 1) :]  # noqa: E203
                    ordered_term += _normal_ordered_term(new_term, coefficient)
            elif left_sub_term[1] == right_sub_term[1]:
                # If indices are same, evaluate it to zero.
                if left_sub_term[0] == right_sub_term[0]:
                    return ordered_term
                # Swap them if same operator but lower index on left
                if left_sub_term[0] < right_sub_term[0]:
                    term[j], term[j - 1] = left_sub_term, right_sub_term

    # Add the term and return.
    ordered_term += QubitExcitationOperator(_qubit_excitation_tuple_to_string(tuple(term)), coefficient)
    return ordered_term


def _qubit_excitation_tuple_to_string(term):
    string = []
    for i in term:
        if i[1] == TermValue.adg:
            string.append(f'{i[0]}^')
        else:
            string.append(str(i[0]))
    return ' '.join(string)


__all__ = ['QubitExcitationOperator']
