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

from ... import mqbackend
from ...core.operators.polynomial_tensor import PolynomialTensor
from ...core.parameterresolver import ParameterResolver
from ._term_value import TermValue
from ._terms_operators import TermsOperator

# NB: C++ actually supports FermionOperatorD and FermionOperatorCD that are purely numerical FermionOperator classes

# ==============================================================================


class FermionOperator(TermsOperator):
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

    cxx_base_klass = mqbackend.FermionOperatorBase
    real_pr_klass = mqbackend.FermionOperatorPRD
    complex_pr_klass = mqbackend.FermionOperatorPRCD

    ensure_complex_coeff = False

    _type_conversion_table = {
        mqbackend.complex_pr: complex_pr_klass,
        complex: complex_pr_klass,
        mqbackend.real_pr: real_pr_klass,
        float: real_pr_klass,
    }

    def __init__(self, terms=None, coefficient=1.0):
        """Initialize a FermionOperator instance."""
        if isinstance(terms, PolynomialTensor):
            terms_ = {}
            for term in terms:
                terms_[tuple((i, TermValue[j]) for i, j in term)] = ParameterResolver(terms[term])
            super().__init__(terms_)
        elif isinstance(terms, mqbackend.FermionOperatorBase):
            super().__init__(terms)
        else:
            super().__init__(terms, coefficient)

    def __str__(self) -> str:
        """Return string expression of a TermsOperator."""
        out = []
        for terms, coeff in self.terms.items():
            terms_str = ' '.join([f"{idx}{'^' if TermValue[fermion] else ''}" for idx, fermion in terms])
            out.append(f"{coeff.expression()} [{terms_str}] ")
        if not out:
            return "0"
        return "+\n".join(out)

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
        return self.__class__(self._cpp_obj.imag)

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
        return self.__class__(self._cpp_obj.real)

    @classmethod
    def from_openfermion(cls, of_ops, dtype=None):
        """
        Convert openfermion fermion operator to mindquantum format.

        Args:
            of_ops (openfermion.FermionOperator): fermion operator from openfermion.
            dtype (type): Type of TermsOperator to generate (ie. real `float` or complex `complex`)
                          NB: this parameter is ignored in the Python version of the QubitOperator

        Returns:
            FermionOperator, fermion operator from mindquantum.
        """
        return super().from_openfermion(of_ops, dtype)

    @classmethod
    def loads(cls, strs: str, dtype: type):
        """
        Load JSON(JavaScript Object Notation) into FermionOperator.

        Args:
            strs (str): The dumped fermion operator string.
            dtype (type): (ignored by this class) Type of QubitOperator to create
                (ie. real, complex, real_pr, complex_pr)

        Returns:
            FermionOperator, the FermionOperator load from strings

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> obj = FermionOperator.loads(f.dumps())
            >>> obj == f
            True
        """
        return super().loads(strs, dtype)

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
        return FermionOperator(self._cpp_obj.normal_ordered())

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
            443
        """
        return self._cpp_obj.dumps(indent)

    def hermitian(self):
        """Return Hermitian conjugate of FermionOperator."""
        return self.__class__(self._cpp_obj.hermitian())

    def matrix(self, n_qubits: int = None):
        """
        Convert this fermion operator to csr_matrix under jordan_wigner mapping.

        Args:
            n_qubits (int): The total qubit of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        return self._cpp_obj.matrix(n_qubits)

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
        for i, j in self._cpp_obj.split():
            yield [ParameterResolver(i), self.__class__(j)]

    # pylint: disable=useless-super-delegation
    def to_openfermion(self):
        """Convert fermion operator to openfermion format."""
        return super().to_openfermion()


__all__ = ['FermionOperator']
