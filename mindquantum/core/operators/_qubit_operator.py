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

"""Original MindQuantum Qubit Operator class."""

import json

import numpy as np
from scipy.sparse import csr_matrix, kron

from ...utils.type_value_check import _check_input_type, _check_int_type
from ..operators._base_operator import _Operator
from ..parameterresolver import ParameterResolver
from ._term_value import TermValue

EQ_TOLERANCE = 1e-8

# Define products of all Pauli operators for symbolic multiplication.
# Note can translate all the lowercase to uppercase 'i'->'I'
_PAULI_OPERATOR_PRODUCTS = {
    (TermValue.I, TermValue.I): (1.0, TermValue.I),
    (TermValue.I, TermValue.X): (1.0, TermValue.X),
    (TermValue.I, TermValue.Y): (1.0, TermValue.Y),
    (TermValue.I, TermValue.Z): (1.0, TermValue.Z),
    (TermValue.X, TermValue.I): (1.0, TermValue.X),
    (TermValue.X, TermValue.X): (1.0, TermValue.I),
    (TermValue.X, TermValue.Y): (1.0j, TermValue.Z),
    (TermValue.X, TermValue.Z): (-1.0j, TermValue.Y),
    (TermValue.Y, TermValue.I): (1.0, TermValue.Y),
    (TermValue.Y, TermValue.X): (-1.0j, TermValue.Z),
    (TermValue.Y, TermValue.Y): (1.0, TermValue.I),
    (TermValue.Y, TermValue.Z): (1.0j, TermValue.X),
    (TermValue.Z, TermValue.I): (1.0, TermValue.Z),
    (TermValue.Z, TermValue.X): (1.0j, TermValue.Y),
    (TermValue.Z, TermValue.Y): (-1.0j, TermValue.X),
    (TermValue.Z, TermValue.Z): (1.0, TermValue.I),
}


def _check_valid_qubit_operator_term(qo_term):
    """Check valid qubit operator term."""
    if qo_term is not None and qo_term != '':
        if not isinstance(qo_term, (str, tuple)):
            raise ValueError(f'Qubit operator requires a string or a tuple, but get {type(qo_term)}')

        operators = (TermValue['X'], TermValue['Y'], TermValue['Z'])
        if isinstance(qo_term, str):
            terms = qo_term.split(' ')
            for term in terms:
                if len(term) < 2 or TermValue[term[0].upper()] not in operators or not term[1:].isdigit():
                    if term:
                        raise ValueError(f'Invalid qubit operator term {term}.')
        if isinstance(qo_term, tuple):
            for term in qo_term:
                operator = TermValue[term[1]] if len(term) > 1 and isinstance(term[1], str) else term[1]
                if len(term) != 2 or not isinstance(term[0], int) or term[0] < 0 or operator not in operators:
                    raise ValueError(f'Invalid qubit operator term {term}.')


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
        dtype (type): (ignored by this class) Type of QubitOperator to create
            (ie. real, complex, real_pr, complex_pr)

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

    __hash__ = None

    def __init__(self, term=None, coefficient=1.0, dtype=None):  # pylint: disable=unused-argument
        """Initialize a QubitOperator object."""
        if isinstance(term, QubitOperator):
            self.operators = term.operators
            self.gates_number = term.gates_number
            self.qubit_type = term.qubit_type
            self.terms = term.terms
        else:
            super().__init__(term, coefficient)
            _check_valid_qubit_operator_term(term)
            self.operators = (TermValue['X'], TermValue['Y'], TermValue['Z'])
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
        """
        Return the gate number when treated in single Hamiltonian.

        Returns:
            int, number of the single qubit quantum gates.
        """
        self.gates_number = 0
        for operator in self.terms:
            n_local_operator = len(operator)
            self.gates_number += n_local_operator
        return self.gates_number

    def to_openfermion(self):
        """Convert qubit operator to openfermion format."""
        # pylint: disable=import-outside-toplevel
        from openfermion import QubitOperator as OFQubitOperator

        terms = {}
        for k, v in self.terms.items():
            if not v.is_const():
                raise ValueError("Cannot convert parameteized fermion operator to openfermion format")
            terms[tuple((t[0], TermValue[t[1]]) for t in k)] = v.const
        qubit_operator = OFQubitOperator()
        qubit_operator.terms = terms
        return qubit_operator

    @staticmethod
    def from_openfermion(of_ops, dtype=None):  # pylint: disable=unused-argument
        """
        Convert qubit operator from openfermion to mindquantum format.

        Args:
            of_ops (openfermion.QubitOperator): Qubit operator from openfermion.
            dtype (type): Type of TermsOperator to generate (ie. real `float` or complex `complex`)
                          NB: this parameter is ignored in the Python version of the QubitOperator

        Returns:
            QubitOperator, qubit operator from mindquantum.
        """
        # pylint: disable=import-outside-toplevel
        from openfermion import QubitOperator as OFQubitOperator

        _check_input_type('of_ops', OFQubitOperator, of_ops)
        qubit_operator = QubitOperator()
        for k, v in of_ops.terms.items():
            qubit_operator.terms[tuple((t[0], TermValue[t[1]]) for t in k)] = ParameterResolver(v)
        return qubit_operator

    def _parse_string(self, terms_string):
        """Parse a term given as a string type.

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
            if TermValue[operator.upper()] not in self.operators:
                raise ValueError(
                    f'Invalid type of operator {operator}.'
                    'The Qubit Pauli operator should be one of this {self.operators}'
                )
            if not index.isdigit() or int(index) < 0:
                raise ValueError(f"Invalid index {self.operators}.The qubit index should be nonnegative integer")

            terms_to_tuple.append((int(index), TermValue[operator]))
            terms_to_tuple = sorted(terms_to_tuple, key=lambda item: item[0])
        return tuple(terms_to_tuple)

    def matrix(self, n_qubits=None):  # pylint: disable=too-many-locals
        """
        Convert this qubit operator to csr_matrix.

        Args:
            n_qubits (int): The total qubits of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
            I,
            X,
            Y,
            Z,
        )

        pauli_map = {
            TermValue.X: csr_matrix(X.matrix().astype(np.complex128)),
            TermValue.Y: csr_matrix(Y.matrix().astype(np.complex128)),
            TermValue.Z: csr_matrix(Z.matrix().astype(np.complex128)),
            TermValue.I: csr_matrix(I.matrix().astype(np.complex128)),
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
                f"Given n_qubits {n_qubits} is small than qubit of qubit operator, which is {n_qubits_local}."
            )
        out = 0
        for term, coeff in self.terms.items():
            if not coeff.is_const():
                raise RuntimeError("Cannot convert a parameterized qubit operator to matrix.")
            coeff = coeff.const
            if not term:
                out += csr_matrix(np.identity(2**n_qubits, dtype=np.complex128)) * coeff
            else:
                tmp = np.array([1], dtype=np.complex128) * coeff
                total = [pauli_map[TermValue['I']] for _ in range(n_qubits)]
                for idx, local_op in term:
                    total[idx] = pauli_map[local_op]
                for i in total:
                    tmp = kron(i, tmp)
                out += tmp
        return out

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
        out = QubitOperator()

        for k, v in self.terms.items():
            out.terms[k] = v.real
        return out

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
        out = QubitOperator()

        for k, v in self.terms.items():
            out.terms[k] = v.imag
        return out

    def _simplify(self, terms, coefficient=1.0):
        r"""Simplify the list by using the commuation and       anti-commutation relationship.

        Args:
            terms (str, list((int, str),), tuple((int, str),)): The input terms_lst could be a sorted list or unsorted
                list e.g. [(3, 'Z'),(2, 'X'), (3,'Z')] -> [(2,'X')]
                Also, it could accept input with tuples ((3, 'Z'),(2, 'X'), (3,'Z')) ->[(2,'X')]
            coefficient (int, float, complex, str, ParameterResolver): The coefficient for the corresponding single
                operators

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
            if left_index == right_index:
                new_coefficient, new_operator = _PAULI_OPERATOR_PRODUCTS[(left_operator, right_operator)]
                left_term = (left_index, new_operator)
                coefficient *= new_coefficient

            else:
                if left_term[1] != TermValue['I']:
                    reduced_terms.append((left_term[0], left_term[1]))
                left_term = right_term
        if left_term[1] != TermValue['I']:
            reduced_terms.append((left_term[0], left_term[1]))
        return coefficient, tuple(reduced_terms)

    def hermitian(self):
        """Return Hermitian conjugate of QubitOperator."""
        conjugate_operator = QubitOperator()
        for term, coefficient in self.terms.items():
            conjugate_operator.terms[term] = coefficient.conjugate()
        return conjugate_operator

    def __str__(self):
        """Return an easy-to-read string representation of the QubitOperator."""
        if not self.terms:
            return '0'

        string_rep = ''
        term_cnt = 0
        for term, coeff in sorted(self.terms.items()):
            term_cnt += 1
            if isinstance(coeff, ParameterResolver):
                tmp_string = f'{coeff.expression()} ['
            else:
                tmp_string = f'{coeff} ['
            # deal with this situation (1,'X') or [1, 'X']
            if term == ():
                tmp_string.join(' ]')
            elif isinstance(term[0], int):
                index, operator = term
                tmp_string += f'{operator}{index} '
            else:
                for sub_term in term:
                    index, operator = sub_term
                    # check validity, if checked before,
                    # then we can take away this step
                    if operator in self.operators:
                        print(f'{str(operator)} {repr(operator)}')
                        tmp_string += f'{operator}{index} '
            if term_cnt < len(self.terms):
                string_rep += f'{tmp_string.strip()}] +\n'
            else:
                string_rep += f'{tmp_string.strip()}] '

        return string_rep

    def __repr__(self):
        """Return a string representation of the object."""
        return str(self)

    def dumps(self, indent=4):
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
        if indent is not None:
            _check_int_type('indent', indent)
        dic = {}
        for term, coeff in self.terms.items():
            string = _qubit_tuple_to_string(term)
            dic[string] = coeff.dumps(indent)
        return json.dumps(dic, indent=indent)

    @staticmethod
    def loads(strs, dtype: type):
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
        _check_input_type('strs', str, strs)
        _check_input_type('dtype', type, dtype)
        dic = json.loads(strs)
        f_op = QubitOperator()
        for k, v in dic.items():
            f_op += QubitOperator(k, ParameterResolver.loads(v))
        return f_op

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
        for i, j in self.terms.items():
            yield [j, QubitOperator(i)]


def _qubit_tuple_to_string(term):
    string = []
    for i in term:
        string.append(f'{i[1]}{i[0]}')
    return ' '.join(string)


__all__ = ['QubitOperator']
