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
This module, we cite and refactor the code in Fermilib and OpenFermion
licensed under Apache 2.0 license.
"""
import copy
from abc import ABCMeta, abstractmethod
import numpy as np

from mindquantum.parameterresolver import ParameterResolver as PR

EQ_TOLERANCE = 1e-8

_validate_coeff_type_num = (int, float, complex, np.int32, np.int64,
                            np.float32, np.float64)
_validate_coeff_type_var = (str, PR, dict)
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
    {'X','Y','Z'}. Note here the Operator capation does not matter here.
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
        """initialize a empty class"""
        if not isinstance(coefficient, _validate_coeff_type):
            raise ValueError(
                "Coefficient must be a numeric type or a string or a ParameterResolver, but get {}."
                .format(type(coefficient)))

        self.terms = {}
        self.operators = None
        self.terms_number = 0
        self.qubit_type = None
        self.coefficient = coefficient
        if isinstance(coefficient, str):
            self.coefficient = PR({self.coefficient: 1})
        if isinstance(coefficient, dict):
            self.coefficient = PR(coefficient)
        if term is None:
            self.terms = {}

    def subs(self, paras):
        """
        replace the symbolical representation with the corresponding
        value.Note the paras are dict type with {"sym1": c1, "sym2":c2}.
        """
        res = copy.deepcopy(self)
        for i in res.terms:
            if isinstance(res.terms[i], PR):
                res.terms[i] = res.terms[i].combination(paras)
        return res

    def _parse_term(self, term):
        """Parse the term whether it is string type or sequence type."""
        if isinstance(term, (tuple, list)):
            term = self._parse_sequence(term)
        elif isinstance(term, str):
            term = self._parse_string(term)
        else:
            raise TypeError('Unexpected term for {}'.format(term))
        return term

    @abstractmethod
    def _parse_string(self, terms_string):
        raise NotImplementedError

    def _parse_sequence(self, terms):
        """parse sequence."""
        if not terms:
            return ()
        if isinstance(terms[0], int):
            self._validate_term(tuple(terms))
            return (terms,)

        for sub_term in terms:
            self._validate_term(sub_term)
        if self.qubit_type:
            terms = sorted(terms, key=lambda term: term[0])
        return tuple(terms)

    def _validate_term(self, term):
        """Check whether the tuple term e.g.(2, 'X') is valid."""
        if len(term) != 2 or not isinstance(term, tuple):
            raise ValueError("Invalid type of format {}".format(term))

        index, operator = term
        if operator not in self.operators:
            raise ValueError(
                "Invalid operator {}. Valid operator should be {}".format(
                    term, self.operators))

        if not isinstance(index, int) or (index < 0):
            raise ValueError(
                'Invalid index {} in term. Index should be non-negative'.
                format(index))

    @abstractmethod
    def _simplify(self, terms, coefficient=1.0):
        raise NotImplementedError

    def __imul__(self, multiplier):
        if isinstance(multiplier, _validate_coeff_type):
            if isinstance(multiplier, str):
                multiplier = PR({multiplier: 1})
            for sub_term in self.terms:
                self.terms[sub_term] *= multiplier
            return self
        # Deal with the same type operators e.g. QubitOperator or
        if isinstance(multiplier, type(self)):
            product_results = {}
            for left_operator, left_coeff in self.terms.items():
                for right_operator, right_coeff in multiplier.terms.items():
                    new_operator = left_operator + right_operator
                    if isinstance(left_coeff, PR) and isinstance(
                            right_coeff, PR):
                        raise ValueError(
                            "Can not multiply two parameterized operator.")
                    new_coeff = left_coeff * right_coeff
                    # Need to simplify the new_operator by using the
                    # commutation and anti_commutation relationship
                    new_coeff, new_terms = self._simplify(
                        new_operator, new_coeff)
                    # Update the product_results:
                    if new_terms in product_results:
                        product_results[new_terms] += new_coeff
                    else:
                        product_results[new_terms] = new_coeff
                    # compress
            self.terms = product_results
            return self
        raise TypeError('Cannot multiply invalid operator type to {}.'.format(
            type(self)))

    def __mul__(self, multiplier):
        if isinstance(multiplier, (*_validate_coeff_type, type(self))):
            product_results = copy.deepcopy(self)
            product_results *= multiplier
            return product_results

        raise TypeError('Cannot multiply invalid operator type to {}.'.format(
            type(self)))

    def __rmul__(self, multiplier):
        if isinstance(multiplier, _validate_coeff_type):
            return self * multiplier

        raise TypeError('Cannot multiply invalid operator type to {}.'.format(
            type(self)))

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float, complex)) and divisor != 0:
            return self * (1. / divisor)
        raise TypeError(
            'Cannot divide the {} by non_numeric type or the divisor is 0.'.
            format(type(self)))

    def __itruediv__(self, divisor):
        if isinstance(divisor, (int, float, complex)) and divisor != 0:
            self *= (1. / divisor)
            return self
        raise TypeError(
            'Cannot divide the {} by non_numeric type or the divisor is 0.'.
            format(type(self)))

    def __neg__(self):
        return self * (-1)

    def __pow__(self, exponent):
        exponential_results = self.__class__('')
        if isinstance(exponent, int) and exponent >= 0:
            for _ in range(exponent):
                exponential_results *= self
            return exponential_results

        raise ValueError(
            'exponent must be a non-negative int, but was {} {}'.format(
                type(exponent), repr(exponent)))

    def __iadd__(self, operator):
        """In-place method for += addition of QubitOperator.

        Args:
            operator (QubitOperator): The operator to add.

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
                if not isinstance(self.terms[term], PR):
                    if abs(self.terms[term]) < EQ_TOLERANCE:
                        self.terms.pop(term)
                else:
                    if self.terms[term].expression() == 0:
                        self.terms.pop(term)
        else:
            raise TypeError('Cannot add invalid operator type to {}.'.format(
                type(self)))

        return self

    def __add__(self, operator):
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
                if not isinstance(self.terms[term], PR):
                    if abs(self.terms[term]) < EQ_TOLERANCE:
                        self.terms.pop(term)
                else:
                    if self.terms[term].expression() == 0:
                        self.terms.pop(term)
        else:
            raise TypeError('Cannot sub invalid operator type to {}.'.format(
                type(self)))

        return self

    def __sub__(self, operator):
        substract_operator = copy.deepcopy(self)
        substract_operator -= operator
        return substract_operator

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            raise TypeError(
                'Cannot compare invalid operator type {} to {}.'.format(
                    type(other), type(self)))
        # terms which are in both Operator:
        tmp_self = self.compress()
        tmp_other = other.compress()
        if not isinstance(tmp_self, type(tmp_other)):
            return False
        for term in set(tmp_self.terms).intersection(set(tmp_other.terms)):
            left = tmp_self.terms[term]
            right = tmp_other.terms[term]
            if not isinstance(left, PR) and not isinstance(right, PR):
                if not abs(left - right) <= max(
                        EQ_TOLERANCE,
                        EQ_TOLERANCE * max(abs(left), abs(right))):
                    return False
                return True
            if isinstance(left, PR) and isinstance(right, PR):
                return left == right
            raise ValueError(
                "Can not compare a parameterized operator with a non parameterized operator."
            )
        # check if there is any term only in one Operator, return false
        for term in set(tmp_self.terms).symmetric_difference(
                set(tmp_other.terms)):
            if term:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def compress(self, abs_tol=EQ_TOLERANCE):
        """Eliminates the very small terms that close to zero.
           Removes small imaginary and real parts.
        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0

        Returns:
            the compressed operator

        Examples:
            >>> ham_compress = QubitOperator('X0 Y3', 0.5) + QubitOperator('X0 Y2', 1e-7)
            >>> ham_compress
            1e-07 [X0 Y2] +
            0.5 [X0 Y3]

            >>> ham_compress.compress(1e-6)
            0.5 [X0 Y3]

            >>> ham_para_compress =  QubitOperator('X0 Y3', 0.5) + QubitOperator('X0 Y2', 'X')
            >>> ham_para_compress
            X [X0 Y2] +
            0.5 [X0 Y3]

            >>> ham_para_compress.compress(1e-7)
            X [X0 Y2] +
            0.5 [X0 Y3]

        """
        new_terms = {}
        for term in self.terms:
            # Remove small imaginary and real parts
            coeff = self.terms[term]
            if not isinstance(coeff, PR):
                if abs(coeff) > abs_tol:
                    if abs(complex(coeff).imag) <= abs_tol:
                        coeff = coeff.real
                    elif abs(complex(coeff).real) <= abs_tol:
                        coeff = 1j * coeff.imag
                    new_terms[term] = coeff
            elif coeff.expression() != 0:
                new_terms[term] = coeff
        self.terms = new_terms
        return self

    @property
    def constant(self):
        """ Returns the value of the constant term.
        """
        if () in self.terms:
            return self.terms[()]
        return 0.0

    @constant.setter
    def constant(self, coefficient):
        """Set the coefficient of the Identity term"""
        self.terms[()] = coefficient
        return self

    def __len__(self):
        return len(self.terms)

    @property
    def size(self):
        """Return the size of the hamiltonian terms"""
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms)

    def __next__(self):
        term, coef = next(self.iter)
        return self.__class__(term, coef)

    def next(self):
        """return the next elements in this Operator"""
        return self.__next__()
