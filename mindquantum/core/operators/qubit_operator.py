# -*- coding: utf-8 -*-
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
"""This is the module for the Qubit Operator. """

import json
import ast
import numpy as np
from scipy.sparse import kron, csr_matrix
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.utils.type_value_check import _check_input_type, _check_int_type
from ._base_operator import _Operator

EQ_TOLERANCE = 1e-8

# Define products of all Pauli operators for symbolic multiplication.
# Note can translate all the lowercase to uppercase 'i'->'I'
_PAULI_OPERATOR_PRODUCTS = {
    ('I', 'I'): (1., 'I'),
    ('I', 'X'): (1., 'X'),
    ('I', 'Y'): (1., 'Y'),
    ('I', 'Z'): (1., 'Z'),
    ('X', 'I'): (1., 'X'),
    ('X', 'X'): (1., 'I'),
    ('X', 'Y'): (1.j, 'Z'),
    ('X', 'Z'): (-1.j, 'Y'),
    ('Y', 'I'): (1., 'Y'),
    ('Y', 'X'): (-1.j, 'Z'),
    ('Y', 'Y'): (1., 'I'),
    ('Y', 'Z'): (1.j, 'X'),
    ('Z', 'I'): (1., 'Z'),
    ('Z', 'X'): (1.j, 'Y'),
    ('Z', 'Y'): (-1.j, 'X'),
    ('Z', 'Z'): (1.0, 'I')
}


def _check_valid_qubit_operator_term(term):
    """Check valid qubit operator term."""
    if term is not None and term != '':
        if not isinstance(term, (str, tuple)):
            raise ValueError('Qubit operator requires a string or a tuple, but get {}'.format(type(term)))

        operators = ('X', 'Y', 'Z')
        if isinstance(term, str):
            terms = term.split(' ')
            for t in terms:
                if len(t) < 2 or t[0].upper() not in operators or not t[1:].isdigit():
                    if t:
                        raise ValueError('Invalid qubit operator term {}.'.format(t))
        if isinstance(term, tuple):
            for t in term:
                if len(t) != 2 or not isinstance(t[0], int) or t[0] < 0 or t[1].upper() not in operators:
                    raise ValueError('Invalid qubit operator term {}.'.format(t))


class QubitOperator(_Operator):
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
        1.0 []
        >>> ham_para = QubitOperator('X0 Y3', 'x')
        >>> ham_para
        x [X0 Y3]
        >>> ham_para.subs({'x':1.2})
        1.2 [X0 Y3]
    """
    __hash__ = None

    def __init__(self, term=None, coefficient=1.0):
        super(QubitOperator, self).__init__(term, coefficient)
        _check_valid_qubit_operator_term(term)
        self.operators = ('X', 'Y', 'Z')
        self.gates_number = 0
        self.qubit_type = True

        if term is not None:
            if term == '':
                term = self._parse_term(())
            else:
                term = self._parse_term(term)
            self.coefficient, term = self._simplify(term, self.coefficient)
            self.terms[term] = self.coefficient

    def count_gates(self):
        """Returns the gate number when treated in single Hamiltonian

        Returns:
            int, number of the single qubit quantum gates.
        """
        self.gates_number = 0
        for operator in self.terms:
            n_local_operator = len(operator)
            self.gates_number += n_local_operator
        return self.gates_number

    def _parse_string(self, terms_string):
        """Parse a term given as a string type

        e.g. For QubitOperator:
                'X2 Y0 Z3' -> ((0, 'Y'),(2, 'X'), (3,'Z'))

        Returns:
            tuple, such as ((0, 'Y'),(2, 'X'), (3,'Z'))

        Raises:
            'XX2' or '1.5 X2' is not the proper format and
            could raise TypeError.
        """
        terms = terms_string.split()
        terms_to_tuple = []
        for sub_term in terms:
            operator = sub_term[0]
            index = sub_term[1:]
            if operator.upper() not in self.operators:
                raise ValueError('Invalid type of operator {}.'
                                 'The Qubit Pauli operator should be one of this {}'.format(operator, self.operators))
            if not index.isdigit() or int(index) < 0:
                raise ValueError("Invalid index {}.The qubit index should be\
                    nonnegative integer".format(self.operators))

            terms_to_tuple.append((int(index), operator))
            terms_to_tuple = sorted(terms_to_tuple, key=lambda item: item[0])
        return tuple(terms_to_tuple)

    def matrix(self, n_qubits=None):
        """
        Convert this qubit operator to csr_matrix.

        Args:
            n_qubits (int): The total qubit of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        from mindquantum import X, Y, Z, I
        pauli_map = {
            'X': csr_matrix(X.matrix().astype(np.complex128)),
            'Y': csr_matrix(Y.matrix().astype(np.complex128)),
            'Z': csr_matrix(Z.matrix().astype(np.complex128)),
            'I': csr_matrix(I.matrix().astype(np.complex128)),
        }
        if not self.terms:
            raise ValueError("Cannot convert empty qubit operator to matrix")
        n_qubits_local = 0
        for term in self.terms:
            for idx, _ in term:
                n_qubits_local = max(n_qubits_local, idx + 1)
        if n_qubits_local == 0 and n_qubits is None:
            raise ValueError("You should specific n_qubits for converting a identity qubit operator.")
        if n_qubits is None:
            n_qubits = n_qubits_local
        _check_int_type("n_qubits", n_qubits)
        if n_qubits < n_qubits_local:
            raise ValueError(
                f"Given n_qubits {n_qubits} is small than qubit of qubit operator, which is {n_qubits_local}.")
        out = 0
        for term, coeff in self.terms.items():
            if isinstance(coeff, PR):
                raise RuntimeError("Cannot convert a parameterized qubit operator to matrix.")
            if not term:
                out += csr_matrix(np.identity(2**n_qubits, dtype=np.complex128)) * coeff
            else:
                tmp = np.array([1], dtype=np.complex128) * coeff
                total = [pauli_map['I'] for _ in range(n_qubits)]
                for idx, s in term:
                    total[idx] = pauli_map[s]
                for i in total:
                    tmp = kron(i, tmp)
                out += tmp
        return out

    @property
    def real(self):
        """
        Convert the coeff to its real part.

        Returns:
            QubitOperator, the real part of this qubit operator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('X0', 1 + 2j) + QubitOperator('Y0', 'a')
            >>> f.real.compress()
            1.0 [X0] +
            a [Y0]
        """
        out = QubitOperator()

        for k, v in self.terms.items():
            out.terms[k] = v.real
        return out

    @property
    def imag(self):
        """
        Convert the coeff to its imag part.

        Returns:
            QubitOperator, the imag part of this qubit operator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('X0', 1 + 2j) + QubitOperator('Y0', 'a')
            >>> f.imag.compress()
            2.0 [X0]
        """
        out = QubitOperator()

        for k, v in self.terms.items():
            out.terms[k] = v.imag
        return out

    def _simplify(self, terms, coefficient=1.0):
        r""" Simplify the list by using the commuation and
        anti-commutation relationship.

        Args:
            terms (str, list((int, str),), tuple((int, str),)): The input terms_lst
                could be a sorted list or unsorted list e.g. [(3, 'Z'),(2, 'X'), (3,'Z')] -> [(2,'X')]
                Also, it could accept input with tuples ((3, 'Z'),(2, 'X'), (3,'Z')) ->[(2,'X')]
            coefficient (int, float, complex, str, ParameterResolver): The coefficient
                for the corresponding single operators

        Returns:
            tuple(coefficient, tuple(reduced_terms)), the simplified coefficient and operators

        """
        if not terms:
            return coefficient, terms
        if isinstance(terms, dict):
            terms = list(terms)
        elif isinstance(terms[0], int):
            return coefficient, tuple(terms)
        else:
            terms = sorted(terms, key=lambda term: term[0])
        reduced_terms = []
        left_term = terms[0]
        for right_term in terms[1:]:
            left_index, left_operator = left_term
            right_index, right_operator = right_term
            left_operator, right_operator = left_operator.upper(), right_operator.upper()
            if left_index == right_index:
                new_coefficient, new_operator = _PAULI_OPERATOR_PRODUCTS[(left_operator, right_operator)]
                left_term = (left_index, new_operator)
                coefficient *= new_coefficient

            else:
                if left_term[1].upper() != 'I':
                    reduced_terms.append((left_term[0], left_term[1].upper()))
                left_term = right_term
        if left_term[1].upper() != 'I':
            reduced_terms.append((left_term[0], left_term[1].upper()))
        return coefficient, tuple(reduced_terms)

    def __str__(self):
        """
        Return an easy-to-read string representation of the QubitOperator.
        """
        if not self.terms:
            return '0'

        string_rep = ''
        term_cnt = 0
        for term, coeff in sorted(self.terms.items()):
            term_cnt += 1
            if isinstance(coeff, PR):
                tmp_string = '{} ['.format(coeff.expression())
            else:
                tmp_string = '{} ['.format(coeff)
            # deal with this situation (1,'X') or [1, 'X']
            if term == ():
                tmp_string.join(' ]')
            elif isinstance(term[0], int):
                index, operator = term
                tmp_string += '{}{} '.format(operator, index)
            else:
                for sub_term in term:
                    index, operator = sub_term
                    # check validity, if checked before,
                    # then we can take away this step
                    if operator in self.operators:
                        tmp_string += '{}{} '.format(operator, index)
            if term_cnt < len(self.terms):
                string_rep += '{}] +\n'.format(tmp_string.strip())
            else:
                string_rep += '{}] '.format(tmp_string.strip())

        return string_rep

    def __repr__(self):
        return str(self)

    def dumps(self, indent=4):
        '''
        Dump QubitOperator into JSON(JavaScript Object Notation)

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            JSON(strings), the JSON strings of QubitOperator

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator('X0 Y1', 1.2) + QubitOperator('Z0 X1', {'a': 2.1})
            >>> print(ops.dumps())
            {
                "((0, 'X'), (1, 'Y'))": "1.2",
                "((0, 'Z'), (1, 'X'))": "{\"a\": 2.1, \"__class__\": \"ParameterResolver\", \
                    \"__module__\": \"parameterresolver.parameterresolver\", \"no_grad_parameters\": []}",
                "__class__": "QubitOperator",
                "__module__": "operators.qubit_operator"
            }
        '''
        if indent is not None:
            _check_int_type('indent', indent)
        d = self.terms

        # Convert key type from tuple into str
        key_list = list(d.keys())
        for i, k in enumerate(key_list):
            key_list[i] = k.__str__()

        # Convert value type from complex/ParameterResolver into str
        value_list = list(d.values())
        for j, v in enumerate(value_list):
            if isinstance(v, (complex, int, float)):
                value_list[j] = str(v)
            elif isinstance(v, PR):
                value_list[j] = (v.dumps(None))
            else:
                raise ValueError("Coefficient must be a complex/int/float type or a ParameterResolver, \
                    but get {}.".format(type(v)))

        dic = dict(zip(key_list, value_list))
        dic['__class__'] = self.__class__.__name__
        dic['__module__'] = self.__module__

        return json.dumps(dic, indent=indent)

    @staticmethod
    def loads(strs):
        '''
        Load JSON(JavaScript Object Notation) into QubitOperator

        Args:
            strs (str): The dumped qubit operator string.

        Returns:
            FermionOperator, the QubitOperator load from strings

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> strings = """{"((0, 'X'), (1, 'Y'))": 1.2, "((0, 'Z'), (1, 'X'))": {"a": 2.1}, \
                "__class__": "QubitOperator", "__module__": "__main__"}"""
            >>> obj = QubitOperator.loads(strings)
            >>> print(obj)
            1.2 [X0 Y1] + 2.1*a [Z0 X1]
        '''
        _check_input_type('strs', str, strs)
        dic = json.loads(strs)
        if '__class__' in dic:
            class_name = dic.pop('__class__')
            if class_name == 'QubitOperator':
                module_name = dic.pop('__module__')
                module = __import__(module_name)
                class_ = getattr(module, class_name)

                # Convert key type from str into tuple
                key_list = list(dic.keys())
                for i, k in enumerate(key_list):
                    key_list[i] = tuple(ast.literal_eval(k))

                # Convert value type from str into ParameterResolver/complex
                value_list = list(dic.values())
                for j, v in enumerate(value_list):
                    if isinstance(v, str):
                        if '__class__' in v:
                            value_list[j] = PR.loads(v)
                        else:
                            value_list[j] = complex(v)

                terms = dict(zip(key_list, value_list))

                q_op = QubitOperator()
                for key, value in terms.items():
                    q_op += class_(key, value)

            else:
                raise TypeError("Require a QubitOperator class, but get {} class".format(class_name))

        else:
            raise ValueError("Expect a '__class__' in strings, but not found")

        return q_op


__all__ = ['QubitOperator']
