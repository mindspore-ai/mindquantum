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
"""
This module serves as the base class for FermionOperator and QubitOperator.

This module, we cite and refactor the code in Fermilib and OpenFermion licensed under Apache 2.0 license.
"""
import copy
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np

from ..parameterresolver import ParameterResolver
from ._term_value import TermValue

EQ_TOLERANCE = 1e-8

_validate_coeff_type_num = (int, float, complex, np.int32, np.int64, np.float32, np.float64)
_validate_coeff_type_var = (str, ParameterResolver, dict)
_validate_coeff_type = (_validate_coeff_type_num, _validate_coeff_type_var)


class _Operator(metaclass=ABCMeta):
    """
    This module serves as the base class for FermionOperator and Operator.

    A operator that store an object which composed of sum of terms;
    Each term is actually a direct product ('index', 'quantum_operator'),
    where 'index' is a non-negative integer
    and the 'quantum_operator' are decided due to its child_class.
    Currently, it refers to FermionOperator and Operator.
    In FermionOperator, it is
    creation operator and annihilation operator,
    which could be represents by  {1: '^', 0: '', '^': '^', '': ''},
    respectively. While for Operator, the 'quantum_operator' is refers to
    Pauli operators, which is the set
    {'X','Y','Z'}. Note here the Operator caption does not matter here.
    {'x','y','z'} is also ok.
    The coefficients of the terms are stored in a dictionary data structure
    who keys are the terms.
    Operators of same type are supported added/subtracted and
    multiplied together.

    Attributes:
        quantum_operator (tuple):
        A tuple of objects representing the possible quantum_operator.
        This should be defined in the subclass.
        for FermionOperator, this is (1, 0) or  ('^', '');
        for Operator, this is ('X','Y','Z')

        quantum_operator_strings (tuple):
        A tuple of string representations of quantum operator.
        These should be in one-to-one correspondence with quantum operator and
        listed in the same order.
        e.g. for FermionOperator, this is ('^', '').

        different_indices_commute (bool): A boolean indicating whether
            factors acting on different indices commute.


        terms (dict):
            **key** (tuple of tuples): A dictionary storing the coefficients
            of the terms in the operator. The keys are the terms.
            A term is a product of individual factors; each factor is
            represented by a tuple of the form (`index`, `action`), and
            these tuples are collected into a larger tuple which represents
            the term as the product of its factors.

    """

    __hash__ = None

    def __init__(self, term=None, coefficient=1.0):
        """Initialize a empty class."""
        if not isinstance(coefficient, _validate_coeff_type):
            raise ValueError(
                f"Coefficient must be a numeric type or a string or a ParameterResolver, but get {type(coefficient)}."
            )

        self.terms = {}
        self.operators = None
        self.terms_number = 0
        self.qubit_type = None
        if isinstance(coefficient, str):
            self.coefficient = ParameterResolver({coefficient: 1})
        elif isinstance(coefficient, dict):
            self.coefficient = ParameterResolver()
            for k, v in coefficient.items():
                self.coefficient[k] = v
        elif isinstance(coefficient, ParameterResolver):
            self.coefficient = coefficient
        elif isinstance(coefficient, numbers.Number):
            self.coefficient = ParameterResolver(coefficient)
        if term is None:
            self.terms = {}

    def subs(self, paras):
        """
        Replace the symbolical representation with the corresponding value.

        Note the paras are dict type with {"sym1": c1, "sym2":c2}.
        """
        res = copy.deepcopy(self)
        for i in res.terms:
            res.terms[i] = res.terms[i].combination(paras)
        return res

    def _parse_term(self, term):
        """Parse the term whether it is string type or sequence type."""
        if isinstance(term, (tuple, list)):
            term = self._parse_sequence(term)
        elif isinstance(term, str):
            term = self._parse_string(term)
        else:
            raise TypeError(f'Unexpected term for {term}')
        return term

    @abstractmethod
    def _parse_string(self, terms_string):
        raise NotImplementedError

    def _parse_sequence(self, terms):
        """Parse a sequence."""
        if not terms:
            return ()
        if isinstance(terms[0], int):
            self._validate_term(tuple(terms))
            return tuple(terms[0], TermValue[terms[1]] if isinstance(terms[1], (str, int)) else terms[1])

        for sub_term in terms:
            self._validate_term(sub_term)
        if self.qubit_type:
            terms = sorted(terms, key=lambda term: term[0])
        return tuple((term[0], TermValue[term[1]] if isinstance(term[1], (str, int)) else term[1]) for term in terms)

    def _validate_term(self, term):
        """Check whether the tuple term e.g.(2, 'X') is valid."""
        if len(term) != 2 or not isinstance(term, tuple):
            raise ValueError(f"Invalid type of format {term}")

        index, operator = term
        if isinstance(operator, str):
            operator = TermValue[operator.upper()]
        if operator not in self.operators:
            raise ValueError(f"Invalid operator {term}. Valid operator should be {self.operators}")

        if not isinstance(index, int) or (index < 0):
            raise ValueError(f'Invalid index {index} in term. Index should be non-negative')

    @abstractmethod
    def _simplify(self, terms, coefficient=1.0):
        raise NotImplementedError

    def __imul__(self, multiplier):
        if isinstance(multiplier, _validate_coeff_type):
            if isinstance(multiplier, str):
                multiplier = ParameterResolver({multiplier: 1})
            for sub_term in self.terms:
                self.terms[sub_term] *= multiplier
            return self
        # Deal with the same type operators e.g. QubitOperator or
        if isinstance(multiplier, type(self)):
            product_results = {}
            for left_operator, left_coeff in self.terms.items():
                for right_operator, right_coeff in multiplier.terms.items():
                    new_operator = left_operator + right_operator
                    new_coeff = left_coeff * right_coeff
                    # Need to simplify the new_operator by using the
                    # commutation and anti_commutation relationship
                    new_coeff, new_terms = self._simplify(new_operator, new_coeff)
                    # Update the product_results:
                    if new_terms in product_results:
                        product_results[new_terms] += new_coeff
                    else:
                        product_results[new_terms] = new_coeff
                    # compress
            self.terms = product_results
            return self
        raise TypeError(f'Cannot multiply invalid operator type to {type(self)}.')

    def __mul__(self, multiplier):
        if isinstance(multiplier, (*_validate_coeff_type, type(self))):
            product_results = copy.deepcopy(self)
            product_results *= multiplier
            return product_results

        raise TypeError(f'Cannot multiply invalid operator type to {type(self)}.')

    def __rmul__(self, multiplier):
        if isinstance(multiplier, _validate_coeff_type):
            return self * multiplier

        raise TypeError(f'Cannot multiply invalid operator type to {type(self)}.')

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float, complex)) and divisor != 0:
            return self * (1.0 / divisor)
        raise TypeError(f'Cannot divide the {type(self)} by non_numeric type or the divisor is 0.')

    def __itruediv__(self, divisor):
        if isinstance(divisor, (int, float, complex)) and divisor != 0:
            self *= 1.0 / divisor
            return self
        raise TypeError(f'Cannot divide the {type(self)} by non_numeric type or the divisor is 0.')

    def __neg__(self):
        return self * (-1)

    def __pow__(self, exponent):
        exponential_results = self.__class__('')
        if isinstance(exponent, int) and exponent >= 0:
            for _ in range(exponent):
                exponential_results *= self
            return exponential_results

        raise ValueError(f'exponent must be a non-negative int, but was {type(exponent)} {repr(exponent)}')

    def __iadd__(self, operator) -> "_Operator":
        """In-place method for += addition of QubitOperator.

        Args:
            operator (QubitOperator, numbers.Number): The operator or a number to add.

        Returns:
            sum (QubitOperator), Mutated self.

        Raises:
            TypeError: Cannot add invalid operator type.
        """
        if isinstance(operator, type(self)):
            for term in operator.terms:
                if term in self.terms:
                    self.terms[term] += operator.terms[term]
                else:
                    self.terms[term] = operator.terms[term]
                if not isinstance(self.terms[term], ParameterResolver):
                    if abs(self.terms[term]) < EQ_TOLERANCE:
                        self.terms.pop(term)
                else:
                    if not self.terms[term]:
                        self.terms.pop(term)
        elif isinstance(operator, numbers.Number):
            self += operator * self.__class__("")
        else:
            raise TypeError(f'Cannot add invalid operator type to {type(self)}.')

        return self

    def __add__(self, operator) -> "_Operator":
        sum_operator = copy.deepcopy(self)
        sum_operator += operator
        return sum_operator

    def __radd__(self, operator) -> "_Operator":
        sum_operator = copy.deepcopy(self)
        sum_operator += operator
        return sum_operator

    def __isub__(self, operator):
        if isinstance(operator, type(self)):
            for term in operator.terms:
                if term in self.terms:
                    self.terms[term] -= operator.terms[term]
                else:
                    self.terms[term] = -operator.terms[term]
                if not isinstance(self.terms[term], ParameterResolver):
                    if abs(self.terms[term]) < EQ_TOLERANCE:
                        self.terms.pop(term)
                else:
                    if not self.terms[term]:
                        self.terms.pop(term)
            return self
        if isinstance(operator, numbers.Number):
            subtract_operator = copy.deepcopy(self)
            subtract_operator -= operator * self.__class__("")
            return subtract_operator
        raise TypeError(f'Cannot sub invalid operator type to {type(self)}.')

    def __sub__(self, operator):
        subtract_operator = copy.deepcopy(self)
        subtract_operator -= operator
        return subtract_operator

    def __rsub__(self, operator):
        return operator + (-1) * self

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            raise TypeError(f'Cannot compare invalid operator type {type(other)} to {type(self)}.')
        # terms which are in both Operator:
        tmp_self = self.compress()
        tmp_other = other.compress()
        if not isinstance(tmp_self, type(tmp_other)):
            return False
        for term in set(tmp_self.terms).intersection(set(tmp_other.terms)):
            left = tmp_self.terms[term]
            right = tmp_other.terms[term]
            if not isinstance(left, ParameterResolver) and not isinstance(right, ParameterResolver):
                if not abs(left - right) <= max(EQ_TOLERANCE, EQ_TOLERANCE * max(abs(left), abs(right))):
                    return False
                return True
            if isinstance(left, ParameterResolver) and isinstance(right, ParameterResolver):
                return left == right
            raise ValueError("Can not compare a parameterized operator with a non parameterized operator.")
        # check if there is any term only in one Operator, return false
        for term in set(tmp_self.terms).symmetric_difference(set(tmp_other.terms)):
            if term:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def compress(self, abs_tol=EQ_TOLERANCE) -> "_Operator":
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
        new_terms = {}
        for term, coeff in self.terms.items():
            # Remove small imaginary and real parts
            if coeff.is_const():
                coeff = coeff.const
                if abs(coeff) > abs_tol:
                    if abs(complex(coeff).imag) <= abs_tol:
                        coeff = coeff.real
                    elif abs(complex(coeff).real) <= abs_tol:
                        coeff = 1j * coeff.imag
                    new_terms[term] = ParameterResolver(coeff)
            elif coeff.expression() != 0:
                new_terms[term] = coeff
        self.terms = new_terms
        return self

    @property
    def constant(self):
        """Return the value of the constant term."""
        if () in self.terms:
            return self.terms[()]
        return 0.0

    @constant.setter
    def constant(self, coefficient):
        """Set the coefficient of the Identity term."""
        self.terms[()] = ParameterResolver(coefficient)
        return self

    def __len__(self):
        return len(self.terms)

    @property
    def size(self):
        """Return the size of the hamiltonian terms."""
        return len(self.terms)

    def __iter__(self):
        for term, coeff in self.terms.items():
            yield self.__class__(term, coeff)

    @property
    def is_singlet(self):
        """
        To verify whether this operator has only one term.

        Returns:
            bool, whether this operator has only one term.
        """
        return len(self.terms) == 1

    def singlet(self):
        """
        Split the single string operator into every word.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Returns:
            list(_Operator): The split word of the string.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator("X0 Y1", 1)
            >>> print(ops.singlet())
            [1 [X0] , 1 [Y1] ]
        """
        if not self.is_singlet:
            raise RuntimeError(f"terms size should be equal to 1, but get {len(self.terms)} terms.")
        words = []
        for term, _ in self.terms.items():
            for k in term:
                words.append(self.__class__((k,), 1))
        return words

    def singlet_coeff(self):
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
            {'a': 1}, const: 0
        """
        if not self.is_singlet:
            raise RuntimeError(f"terms size should be equal to 1, but get {len(self.terms)} terms.")
        for i in self.terms.values():
            return i
        return None
