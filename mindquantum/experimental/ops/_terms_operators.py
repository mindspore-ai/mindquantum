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

from abc import ABCMeta, abstractmethod
import numbers
from typing import Dict, Tuple, Union

from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.mqbackend import complex_pr, real_pr
from mindquantum.utils.type_value_check import _check_input_type

from ...core._arithmetic_ops_adaptor import CppArithmeticAdaptor
from .._mindquantum_cxx.ops import (
    EQ_TOLERANCE,
    CmplxPRSubsProxy,
    DoublePRSubsProxy,
)
from .. import TermValue


class TermsOperator(CppArithmeticAdaptor, metaclass=ABCMeta):
    """Abstract base class for terms operators (FermionOperator and QubitOperator)."""

    cxx_base_klass: type
    real_pr_klass: type
    complex_pr_klass: type
    openfermion_klass: type

    @classmethod
    @abstractmethod
    def create_cpp_obj(cls, term, coeff):
        """
        Create a new instance of a C++ child class.

        Args:
            term (str): The input term of qubit operator. Default: None.
            coeff (Union[numbers.Number, str, ParameterResolver]): The coefficient of this qubit operator, could be a
                number or a variable represent by a string or a symbol or a parameter resolver.
                Default: 1.0.
        """
        pass

    def __init__(self, term=None, coeff=1.0):
        """Initialize a TermsOperator object."""
        if isinstance(term, self.cxx_base_klass):
            self._cpp_obj = term
        else:
            self._cpp_obj = self.__class__.create_cpp_obj(term, coeff)

    def __copy__(self, memodict) -> 'TermsOperator':
        """Deep copy this TermsOperator."""
        return self.__class__(self._cpp_obj.__copy__())

    def __deepcopy__(self, memodict) -> 'TermsOperator':
        """Deep copy this TermsOperator."""
        return self.__class__(self._cpp_obj.__copy__())

    def __repr__(self) -> str:
        """Return string expression of a TermsOperator."""
        return self.__str__()

    def __iter__(self):
        """Iterate every single term."""
        for term, coeff in self.terms.items():
            yield self.__class__(term, coeff)

    def __len__(self) -> int:
        """Return the size of term."""
        return self._cpp_obj.size

    @property
    def imag(self):
        """
        Convert the coefficient to its imag part.

        Returns:
            SubClass, the imag part of this TermsOperator.

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
            SubClass, the real part of this TermsOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.real.compress()
            1.0 [0] +
            a [0^]
        """
        return self.__class__(self._cpp_obj.real)

    @property
    def terms(self) -> Dict[Tuple[Tuple[int]], ParameterResolver]:
        """Get the terms of a TermsOperator."""
        return {tuple(i): ParameterResolver(j) for i, j in self._cpp_obj.terms()}

    def compress(self, abs_tol=EQ_TOLERANCE):
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
        return self.__class__(self._cpp_obj.compress(abs_tol))

    @property
    def constant(self) -> ParameterResolver:
        """Return the value of the constant term."""
        return ParameterResolver(self._cpp_obj.constant)

    @constant.setter
    def constant(self, coeff: Union[float, complex, ParameterResolver]):
        """Set the coefficient of the Identity term."""
        self._cpp_obj.constant = coeff  # C++ handles the conversion from coeff to ParameterResolver

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
        return self._cpp_obj.count_qubits()

    def dumps(self, indent: int = 4) -> str:
        r"""
        Dump a TermsOperator into JSON(JavaScript Object Notation).

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            JSON (str), the JSON strings of this TermsOperator

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> len(f.dumps())
            922
        """
        return self._cpp_obj.dumps(indent)

    @classmethod
    def loads(cls, strs: str, dtype: type):
        """
        Load JSON(JavaScript Object Notation) into a TermsOperator.

        Args:
            strs (str): The dumped fermion operator string.
            dtype (type): Type of coefficient for the resulting TermsOperator

        Returns:
            SubClass, the SubClass loaded from JSON-formatted strings

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> obj = FermionOperator.loads(f.dumps())
            >>> obj == f
            True
        """
        if dtype == real_pr or dtype == float:
            klass = cls.real_pr_klass
        elif dtype == complex_pr or dtype == complex:
            klass = cls.complex_pr_klass
        else:
            raise TypeError(f'Unsupported dtype ({dtype})!')

        return cls(klass.loads(strs))

    def get_coeff(self, term):
        """Get coefference of given term."""
        return ParameterResolver(self._cpp_obj.get_coeff(term))

    def hermitian(self):
        """
        Get the hermitian of a TermsOperator.

        Returns:
            The hermitian of this TermsOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> a = FermionOperator("0^ 1", {"a": 1 + 2j})
            >>> a.hermitian()
            (1-2j)*a [1^ 0]
        """
        return self.__class__(self._cpp_obj.hermitian())

    def matrix(self, n_qubits: int = None):
        """
        Convert this fermion operator to csr_matrix under jordan_wigner mapping.

        Args:
            n_qubits (int): The total qubit of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        return self._cpp_obj.matrix(n_qubits)

    def subs(self, params_value: ParameterResolver):
        """Replace the symbolical representation with the corresponding value."""
        if isinstance(params_value, dict):
            params_value = ParameterResolver(params_value)
        if isinstance(self._cpp_obj, self.real_pr_klass):
            return self.__class__(self._cpp_obj.subs(DoublePRSubsProxy(params_value._cpp_obj)))
        return self.__class__(self._cpp_obj.subs(CmplxPRSubsProxy((params_value._cpp_obj.to_complex()))))

    @property
    def is_singlet(self) -> bool:
        """
        To verify whether this operator has only one term.

        Returns:
            bool, whether this operator has only one term.
        """
        return self._cpp_obj.is_singlet

    def singlet(self):
        """
        Split the single string operator into every word.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Returns:
            List[SubClass]: The split word of the string.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> ops = FermionOperator("1^ 2", 1)
            >>> print(ops.singlet())
            [1 [1^] , 1 [2] ]
        """
        return [self.__class__(i) for i in self._cpp_obj.singlet()]

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
        return ParameterResolver(self._cpp_obj.singlet_coeff())

    @property
    def size(self):
        """Return the number of terms of this TermsOperator."""
        return self._cpp_obj.size

    # TODO(xusheng): Finish type hint.
    def split(self):
        """
        Split the coefficient and the operator.

        Returns:
            List[List[ParameterResolver, TermsOperator]], the split result.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> a = FermionOperator('0', 'a') + FermionOperator('1^', 1.2)
            >>> list(a.split())
            [[{'a': 1}, const: 0, 1 [0] ], [{}, const: 1.2, 1 [1^] ]]
        """
        for i, j in self._cpp_obj.split():
            yield [ParameterResolver(i), self.__class__(j)]

    def to_openfermion(self):
        """Convert a TermsOperator openfermion format."""
        terms = {}
        for term, pr in self.terms.items():
            if not pr.is_const:
                raise ValueError(f'Cannot convert parameterized {self.__class__.name} to OpenFermion format')
            terms[tuple((i, TermValue[j]) for i, j in term)] = pr.const

        operator = self.__class__.openfermion_klass()
        operator.terms = terms
        return operator

    @classmethod
    def from_openfermion(cls, of_ops):
        """
        Convert openfermion fermion operator to mindquantum format.

        Args:
            of_ops: fermion operator from openfermion.

        Returns:
            TermsOperator, terms operator from mindquantum.
        """
        _check_input_type('of_ops', cls.openfermion_klass, of_ops)
        klass = cls.real_pr_klass
        c = 1
        for v in of_ops.terms.values():
            if isinstance(v, numbers.Complex) and not isinstance(v, numbers.Real):
                klass = cls.complex_pr_klass
                c = 1 + 0j
                break
        list_terms = []
        list_coeff = []
        for k, v in of_ops.terms.items():
            list_terms.append(tuple((i, TermValue[j]) for i, j in k))
            list_coeff.append((ParameterResolver(v) * c)._cpp_obj)
        cpp_obj = klass(list_terms, list_coeff)
        return cls(cpp_obj)
