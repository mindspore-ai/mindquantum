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

from ... import mqbackend
from ...core.parameterresolver import ParameterResolver
from ._term_value import TermValue
from ._terms_operators import TermsOperator

# NB: C++ actually supports FermionOperatorD and FermionOperatorCD that are purely numerical FermionOperator classes

# ==============================================================================


class QubitOperator(TermsOperator):
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

    # NB: In principle, we support real-valued qubit operators. Unfortunately, in this case, not all coefficient
    #     simplifications are possible.
    #     For now, we simply force any Python code to create complex qubit operators...

    cxx_base_klass = mqbackend.QubitOperatorBase
    real_pr_klass = mqbackend.QubitOperatorPRD
    complex_pr_klass = mqbackend.QubitOperatorPRCD

    ensure_complex_coeff = True

    _type_conversion_table = {
        mqbackend.complex_pr: complex_pr_klass,
        complex: complex_pr_klass,
        mqbackend.real_pr: complex_pr_klass,
        float: complex_pr_klass,
    }

    def __init__(self, terms=None, coefficient=1.0):
        """Initialize a QubitOperator instance."""
        if isinstance(terms, mqbackend.QubitOperatorBase):
            super().__init__(terms)
        else:
            super().__init__(terms, coefficient)

    def __str__(self) -> str:
        """Return string expression of a TermsOperator."""
        out = []
        for terms, coeff in self.terms.items():
            terms_str = ' '.join([f"{TermValue[pauli]}{idx}" for idx, pauli in terms])
            out.append(f"{coeff.expression()} [{terms_str}] ")
        if not out:
            return "0"
        return "+\n".join(out)

    @property
    def imag(self):
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
        return self.__class__(self._cpp_obj.imag)

    @property
    def real(self):
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
        return self.__class__(self._cpp_obj.real)

    @classmethod
    def from_openfermion(cls, of_ops, dtype=None):
        """
        Convert qubit operator from openfermion to mindquantum format.

        Args:
            of_ops (openfermion.QubitOperator): Qubit operator from openfermion.
            dtype (type): Type of TermsOperator to generate (ie. real `float` or complex `complex`)
                          NB: this parameter is ignored in the Python version of the QubitOperator

        Returns:
            QubitOperator, qubit operator from mindquantum.
        """
        return super().from_openfermion(of_ops, dtype)

    @classmethod
    def loads(cls, strs: str, dtype: type):
        """
        Load JSON(JavaScript Object Notation) into QubitOperator.

        Args:
            strs (str): The dumped qubit operator string.
            dtype (type): (ignored by this class) Type of QubitOperator to create
                (ie. real, complex, real_pr, complex_pr)

        Returns:
            QubitOperator`, the QubitOperator load from strings

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator('X0 Y1', 1.2) + QubitOperator('Z0 X1', {'a': 2.1})
            >>> obj = QubitOperator.loads(ops.dumps())
            >>> obj == ops
            True
        """
        return super().loads(strs, dtype)

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
        return self._cpp_obj.count_gates()

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
            448
        """
        return self._cpp_obj.dumps(indent)

    def hermitian(self):
        """Return Hermitian conjugate of QubitOperator."""
        return self.__class__(self._cpp_obj.hermitian())

    def matrix(self, n_qubits: int = None):
        """
        Convert this qubit operator to csr_matrix.

        Args:
            n_qubits (int): The total qubits of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        return self._cpp_obj.matrix(n_qubits)

    def split(self):
        """
        Split the coefficient and the operator.

        Returns:
            List[List[ParameterResolver, QubitOperator]], the split result.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> a = QubitOperator('X0', 'a') + QubitOperator('Z1', 1.2)
            >>> list(a.split())
            [[{'a': 1}, const: 0, 1 [X0] ], [{}, const: 1.2, 1 [Z1] ]]
        """
        for i, j in self._cpp_obj.split():
            yield [ParameterResolver(i), self.__class__(j)]

    # pylint: disable=useless-super-delegation
    def to_openfermion(self):
        """Convert qubit operator to openfermion format."""
        return super().to_openfermion()


__all__ = ['QubitOperator']
