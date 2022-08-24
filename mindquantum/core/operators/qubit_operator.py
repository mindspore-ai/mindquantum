#   Portions Copyright (c) 2020 Huawei Technologies Co.,ltd.
#   Portions Copyright 2017 The OpenFermion Developers.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   This module we develop is default being licensed under Apache 2.0 license,
#   and also uses or refactor Fermilib and OpenFermion licensed under
#   Apache 2.0 license.
"""This is the module for the Qubit Operator."""
from typing import Dict, List, Tuple


from mindquantum.core.operators._base_operator import EQ_TOLERANCE
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.experimental import TermValueStr
from mindquantum.experimental._mindquantum_cxx.ops import (
    QubitOperatorPR as QubitOperator_,
)
from mindquantum.experimental.utils import TermValueCpp
from mindquantum.utils.type_value_check import _check_input_type


class QubitOperator(QubitOperator_):
    """
    A sum of terms acting on qubits, e.g., 0.5 * 'X1 X5' + 0.3 * 'Z1 Z2'.

    A term is an operator acting on n qubits and can be represented as:
    coefficient * local_operator[0] x ... x local_operator[n-1]
    where x is the tensor product. A local operator is a Pauli operator
    ('I', 'X', 'Y', or 'Z') which acts on one qubit. In mathematical notation
    a QubitOperator term is, for example, 0.5 * 'X1 X5', which means that a Pauli X operator acts
    on qubit 1 and 5, while the identity operator acts on all the rest qubits.

    Note that a Hamiltonian composed of QubitOperators should be a hermitian
    operator, thus requires the coefficients of all terms must be real.

    QubitOperator has the following attributes set as follows:
    operators = ('X', 'Y', 'Z'), different_indices_commute = True.

    Args:
        term (str): The input term of qubit operator. Default: None.
        coefficient (Union[numbers.Number, str, ParameterResolver]): The
            coefficient of this qubit operator, could be a number or a variable
            represent by a string or a symbol or a parameter resolver. Default: 1.0.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator
        >>> ham = ((QubitOperator('X0 Y3', 0.5)
        ...         + 0.6 * QubitOperator('X0 Y3')))
        >>> ham2 = QubitOperator('X0 Y3', 0.5)
        >>> ham2 += 0.6 * QubitOperator('X0 Y3')
        >>> ham2
        1.1 [X0 Y3]
        >>> ham3 = QubitOperator('')
        >>> ham3
        1 []
        >>> ham_para = QubitOperator('X0 Y3', 'x')
        >>> ham_para
        x [X0 Y3]
        >>> ham_para.subs({'x':1.2})
        6/5 [X0 Y3]
    """

    def __init__(self, term=None, coeff=1.0):
        """Initialize a QubitOperator object."""
        if isinstance(term, QubitOperator_):
            QubitOperator_.__init__(self, term)
        else:
            if term is None:
                QubitOperator_.__init__(self)
            if isinstance(term, dict):
                QubitOperator_.__init__(self, term)
            else:
                if isinstance(term, str):
                    term = term.upper()
                if not isinstance(coeff, ParameterResolver):
                    coeff = ParameterResolver(coeff)
                QubitOperator_.__init__(self, term, coeff)

    def __deepcopy__(self, memodict) -> "QubitOperator":
        """Deep copy this QubitOperator."""
        return QubitOperator(self)

    def __str__(self) -> str:
        """Return string expression of QubitOperator."""
        terms = self.terms
        new_str = ''
        for idx, (term, coeff) in enumerate(terms.items()):
            term = [f"{TermValueStr[j]}{i}" for i, j in term]
            end = ' +\n'
            if idx == len(terms) - 1:
                end = ' '
            new_str += f"{coeff.expression()} [{' '.join(term)}]{end}"
        return new_str if new_str else "0"

    def __repr__(self) -> str:
        """Return string expression of QubitOperator."""
        return self.__str__()

    def __iter__(self) -> "QubitOperator":
        """Iterate every single term."""
        for term, coeff in self.terms.items():
            yield QubitOperator(term, coeff)

    def __len__(self) -> int:
        """Return the size of term."""
        return self.size

    def __neg__(self) -> "QubitOperator":
        """Return negative QubitOperator."""
        return QubitOperator(QubitOperator_.__neg__(self))

    def __add__(self, other) -> "QubitOperator":
        """Add a number or a QubitOperator."""
        return QubitOperator(QubitOperator_.__add__(self, other))

    def __iadd__(self, other) -> "QubitOperator":
        """Inplace add a number or a QubitOperator."""
        QubitOperator_.__iadd__(self, other)
        return self

    def __radd__(self, other) -> "QubitOperator":
        """Right add a number or a QubitOperator."""
        return QubitOperator(QubitOperator_.__add__(self, other))

    def __sub__(self, other) -> "QubitOperator":
        """Subtract a number or a QubitOperator."""
        return QubitOperator(QubitOperator_.__sub__(self, other))

    def __isub__(self, other) -> "QubitOperator":
        """Inplace subtrace a number or a QubitOperator."""
        QubitOperator_.__isub__(self, other)
        return self

    def __rsub__(self, other) -> "QubitOperator":
        """Subtrace a number or a QubitOperator this QubitOperator."""
        return other + (-self)

    def __mul__(self, other) -> "QubitOperator":
        """Multiple a number or a QubitOperator."""
        if isinstance(other, str):
            other = ParameterResolver(other)
        return QubitOperator(QubitOperator_.__mul__(self, other))

    def __imul__(self, other) -> "QubitOperator":
        """Inplace multiple a number or a QubitOperator."""
        if isinstance(other, str):
            other = ParameterResolver(other)
        QubitOperator_.__imul__(self, other)
        return self

    def __rmul__(self, other) -> "QubitOperator":
        """Right multiple a number or a QubitOperator."""
        if isinstance(other, str):
            other = ParameterResolver(other)
        return QubitOperator(QubitOperator_.__mul__(self, other))

    def __truediv__(self, other) -> "QubitOperator":
        """Divide a number."""
        return QubitOperator(QubitOperator_.__truediv__(self, other))

    def __itruediv__(self, other) -> "QubitOperator":
        """Divide a number."""
        return QubitOperator(QubitOperator_.__itruediv__(self, other))

    def __power__(self, exponent: int) -> "QubitOperator":
        """Exponential of QubitOperator."""
        return QubitOperator(QubitOperator_.__power__(self, exponent))

    def __eq__(self, other) -> bool:
        """Check whether two QubitOperator equal."""
        return QubitOperator_.__eq__(self, other)

    def __ne__(self, other) -> bool:
        """Check whether two QubitOperator not equal."""
        return not QubitOperator_.__eq__(self, other)

    @property
    def imag(self) -> "QubitOperator":
        """
        Convert the coefficient to its imag part.

        Returns:
            QubitOperator, the imag part of this qubit operator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('X0', 1 + 2j) + QubitOperator('Y0', 'a')
            >>> f.imag.compress()
            2 [X0]
        """
        return QubitOperator(QubitOperator_.imag(self))

    @property
    def real(self) -> "QubitOperator":
        """
        Convert the coefficient to its real part.

        Returns:
            QubitOperator, the real part of this qubit operator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('X0', 1 + 2j) + QubitOperator('Y0', 'a')
            >>> f.real.compress()
            1 [X0] +
            a [Y0]
        """
        return QubitOperator(QubitOperator_.real(self))

    @property
    def terms(self) -> Dict[Tuple[Tuple[int]], ParameterResolver]:
        """Get the term of QubitOperator."""
        return {tuple(i): ParameterResolver(j) for i, j in QubitOperator_.terms(self)}

    def compress(self, abs_tol=EQ_TOLERANCE) -> "QubitOperator":
        """
        Eliminate the very small terms that close to zero.

        Removes small imaginary and real parts.

        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0

        Returns:
            the compressed operator

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ham_compress = QubitOperator('X0 Y3', 0.5) + QubitOperator('X0 Y2', 1e-7)
            >>> ham_compress
            1/10000000 [X0 Y2] +
            1/2 [X0 Y3]
            >>> ham_compress.compress(1e-6)
            1/2 [X0 Y3]
            >>> ham_para_compress =  QubitOperator('X0 Y3', 0.5) + QubitOperator('X0 Y2', 'X')
            >>> ham_para_compress
            X [X0 Y2] +
            1/2 [X0 Y3]
            >>> ham_para_compress.compress(1e-7)
            X [X0 Y2] +
            1/2 [X0 Y3]
        """
        return QubitOperator(QubitOperator_.compress(self, abs_tol))

    @property
    def constant(self) -> "QubitOperator":
        """Return the value of the constant term."""
        return QubitOperator_.constant(self)

    @constant.setter
    def constant(self, coeff):
        """Set the coefficient of the Identity term."""
        if not isinstance(coeff, ParameterResolver):
            coeff = ParameterResolver(coeff)
        QubitOperator_.constant(self, coeff)

    def count_gates(self):
        """
        Return the gate number when treated in single Hamiltonian.

        Returns:
            int, number of the single qubit quantum gates.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> a = QubitOperator("X0 Y1") + QubitOperator("X2 Z3")
            >>> a.count_gates()
            4
        """
        return QubitOperator_.count_gates(self)

    def count_qubits(self) -> int:
        """
        Calculate the number of qubits on which operator acts before removing the unused qubit.

        Returns:
            int, the qubits number before remove unused qubit.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> a = QubitOperator("X0 Y3")
            >>> a.count_qubits()
            4
        """
        return QubitOperator_.count_qubits(self)

    def dumps(self, indent: int = 4) -> str:
        r"""
        Dump QubitOperator into JSON(JavaScript Object Notation).

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            JSON(strings), the JSON strings of QubitOperator

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator('X0 Y1', 1.2) + QubitOperator('Z0 X1', {'a': 2.1})
            >>> len(ops.dumps())
            1090
        """
        return QubitOperator_.dumps(self, indent)

    @staticmethod
    def loads(strs: str) -> "QubitOperator":
        """
        Load JSON(JavaScript Object Notation) into QubitOperator.

        Args:
            strs (str): The dumped qubit operator string.

        Returns:
            QubitOperator`, the QubitOperator load from strings

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator('X0 Y1', 1.2) + QubitOperator('Z0 X1', {'a': 2.1})
            >>> obj = QubitOperator.loads(ops.dumps())
            >>> obj == ops
            True
        """
        return QubitOperator(QubitOperator_.loads(strs))

    def hermitian(self) -> "QubitOperator":
        """
        Get the hermitian of QubitOperator.

        Returns:
            The hermitian of QubitOperator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> a = QubitOperator("X0 Z2", {"a": 1 + 2j})
            >>> a.hermitian()
            (1-2j)*a [X0 Z2]
        """
        return QubitOperator(QubitOperator_.hermitian(self))

    def matrix(self, n_qubits: int = None):
        """
        Convert this qubit operator to csr_matrix.

        Args:
            n_qubits (int): The total qubits of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        if n_qubits is None:
            n_qubits = self.count_qubits()
        return QubitOperator(QubitOperator_.matrix(self, n_qubits))

    def subs(self, params_value: ParameterResolver) -> "QubitOperator":
        """Replace the symbolical representation with the corresponding value."""
        if not isinstance(params_value, ParameterResolver):
            params_value = ParameterResolver(params_value)
        return QubitOperator(QubitOperator_.subs(self, params_value))

    @property
    def is_singlet(self) -> bool:
        """
        To verify whether this operator has only one term.

        Returns:
            bool, whether this operator has only one term.
        """
        return QubitOperator_.is_singlet(self)

    def singlet(self) -> List["QubitOperator"]:
        """
        Split the single string operator into every word.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Returns:
            List[QubitOperator]: The split word of the string.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator("X1 Z2", 1)
            >>> print(ops.singlet())
            [1 [X1] , 1 [Z2] ]
        """
        return [QubitOperator(i) for i in QubitOperator_.singlet(self)]

    def singlet_coeff(self) -> ParameterResolver:
        """
        Get the coefficient of this operator, if the operator has only one term.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Returns:
            ParameterResolver: the coefficient of this single string operator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator("X0 Y1", "a")
            >>> print(ops.singlet_coeff())
            {'a': (1,0)}, const: (0,0)
        """
        return ParameterResolver(QubitOperator_.singlet_coeff(self))

    @property
    def size(self):
        """Return the size of the QubitOperator terms."""
        return QubitOperator_.size(self)

    # TODO(xusheng): Finish type hint.
    def split(self):
        """
        Split the coefficient and the operator.

        Returns:
            List[List[ParameterResolver, QubitOperator]], the split result.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> a = QubitOperator('X0', 'a') + QubitOperator('Z1', 1.2)
            >>> list(a.split())
            [[{'a': (1,0)}, const: (0,0), 1 [X0] ], [{}, const: (1.2,0), 1 [Z1] ]]
        """
        for i, j in QubitOperator_.split(self):
            yield [ParameterResolver(i), QubitOperator(j)]

    def to_openfermion(self):
        """Convert qubit operator to openfermion format."""
        # pylint: disable=import-outside-toplevel
        from openfermion import QubitOperator as OFQubitOperator

        terms = {}
        for term, pr in self.terms.items():
            if not pr.is_const:
                raise ValueError("Cannot convert parameteized fermion operator to openfermion format")
            terms[tuple((i, TermValueStr[j]) for i, j in term)] = pr.const
        fermion_operator = OFQubitOperator()
        fermion_operator.terms = terms
        return fermion_operator

    @staticmethod
    def from_openfermion(of_ops):
        """
        Convert qubit operator from openfermion to mindquantum format.

        Args:
            of_ops (openfermion.QubitOperator): Qubit operator from openfermion.

        Returns:
            QubitOperator, qubit operator from mindquantum.
        """
        # pylint: disable=import-outside-toplevel
        from openfermion import QubitOperator as OFQubitOperator

        _check_input_type('of_ops', OFQubitOperator, of_ops)
        terms = {}
        for k, v in of_ops.terms.items():
            terms[tuple((i, TermValueCpp[j]) for i, j in k)] = ParameterResolver(v)
        return QubitOperator(terms)


def _qubit_tuple_to_string(term):
    string = []
    for i in term:
        string.append(f'{i[1]}{i[0]}')
    return ' '.join(string)


__all__ = ['QubitOperator']

if __name__ == '__main__':
    ops = QubitOperator("X0 Y1", "a")
