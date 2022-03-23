# -*- coding: utf-8 -*-
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
"""This module is generated the Fermion Operator"""

import json
import ast
from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import kron
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.utils.type_value_check import _check_input_type, _check_int_type
from ._base_operator import _Operator


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
    """n_identity"""
    if n == 0:
        return csr_matrix(np.array([1]), dtype=np.complex128)
    tmp = [csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.complex128)) for _ in range(n)]
    for i in tmp[1:]:
        tmp[0] = kron(tmp[0], i)
    return tmp[0]


@lru_cache()
def _single_fermion_word(idx, dag, n_qubits):
    """single_fermion_word"""
    m = csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.complex128))
    if dag:
        m = csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.complex128))
    return kron(_n_identity(n_qubits - 1 - idx), kron(m, _n_sz(idx)))


@lru_cache()
def _two_fermion_word(idx1, dag1, idx2, dag2, n_qubits):
    """two_fermion_word"""
    return _single_fermion_word(idx1, dag1, n_qubits) * _single_fermion_word(idx2, dag2, n_qubits)


def _check_valid_fermion_operator_term(term):
    """Check valid fermion operator term."""
    if term is not None and term != '':
        if not isinstance(term, (str, tuple)):
            raise ValueError('Fermion operator requires a string or a tuple, but get {}'.format(type(term)))
        if isinstance(term, str):
            terms = term.split(' ')
            for t in terms:
                if (t.endswith('^') and not t[:-1].isdigit()) or (not t.endswith('^') and not t.isdigit()):
                    if t:
                        raise ValueError('Invalid fermion operator term {}'.format(t))
        if isinstance(term, tuple):
            for t in term:
                if len(t) != 2 or not isinstance(t[0], int) or not isinstance(t[1], int) or t[0] < 0 or t[1] not in [
                        0, 1
                ]:
                    raise ValueError('Invalid fermion operator term {}'.format(t))


class FermionOperator(_Operator):
    r"""
    The Fermion Operator such as FermionOperator(' 4^ 3 9 3^ ')
    are used to represent :math:`a_4^\dagger a_3 a_9 a_3^\dagger`.
    These are the Basic Operators to describe a fermionic system,
    such as a Molecular system.
    The FermionOperator are follows the anti-commutation relationship.

    Args:
        terms (str): The input term of fermion operator. Default: None.
        coefficient (Union[numbers.Number, str, ParameterResolver]): The
            coefficient for the corresponding single operators Default: 1.0.

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

    __hash__ = None

    def __init__(self, term=None, coefficient=1.0):
        super(FermionOperator, self).__init__(term, coefficient)
        _check_valid_fermion_operator_term(term)
        self.operators = {1: '^', 0: '', '^': '^', '': ''}
        self.gates_number = 0
        self.qubit_type = False

        if term is not None:
            if term == '':
                term = self._parse_term(())
            else:
                term = self._parse_term(term)
            self.terms[term] = self.coefficient

    def _simplify(self, terms, coefficient=1.0):
        """Simplify a term."""
        return coefficient, tuple(terms)

    def _parse_string(self, terms_string):
        """
        Parse a term given as a string type
        e.g. For FermionOperator:
                 4^ 3  -> ((4, 1),(3, 0))
        Note here the '1' and '0' in the second col represents creation
        and annihilaiton operator respectively

        Returns:
            tuple, return a tuple list, such as ((4, 1),(3, 0))

        Raises:
            '1.5 4^ 3' is not the proper format and
            could raise TypeError.
        """
        map_operator_to_integer_rep = lambda operator: 1 if operator == '^' else 0
        terms = terms_string.split()
        terms_to_tuple = []
        for sub_term in terms:
            index = int(sub_term[0])
            operator = sub_term[1:]
            # Handle such cases: 10^, 100^, ...
            if len(sub_term) >= 2:
                if '^' in sub_term:
                    operator = '^'
                    index = int(sub_term[:sub_term.index(operator)])
                else:
                    operator = ''
                    index = int(sub_term)

            if operator not in self.operators:
                raise ValueError('Invalid type of operator {}.'
                                 'The Fermion operator should be one of this {}'.format(operator, self.operators))
            if index < 0:
                raise ValueError("Invalid index {}.The qubit index should be\
                    non negative integer".format(self.operators))
            terms_to_tuple.append((index, map_operator_to_integer_rep(operator)))
            # check the commutate terms with same index in the list and
            # replace it with the corresponding commutation relationship
        return tuple(terms_to_tuple)

    def __str__(self):
        """
        Return an easy-to-read string representation of the FermionOperator.
        """
        if not self.terms:
            return '0'
        string_rep = ''
        term_cnt = 0
        for term, coeff in sorted(self.terms.items()):
            term_cnt += 1
            if isinstance(coeff, PR):
                tmp_string = '{} ['.format(coeff.expression())  # begin of the '['
            else:
                tmp_string = '{} ['.format(coeff)  # begin of the '['
            # deal with this situation (1,'X') or [1, 'X']
            if term == ():
                if self.size == 1:
                    tmp_string.join(' ]')
                else:
                    pass

            elif isinstance(term[0], int):
                index, operator = term
                if operator in self.operators:
                    tmp_string += '{}{} '.format(index, self.operators[operator])
            else:
                for sub_term in term:
                    index, operator = sub_term
                    # check validity, if checked before,
                    # then we can take away this step
                    if operator in self.operators:
                        tmp_string += '{}{} '.format(index, self.operators[operator])

            if term_cnt < len(self.terms):
                string_rep += '{}] +\n'.format(tmp_string.strip())  # end of the ']'
            else:
                string_rep += '{}] '.format(tmp_string.strip())  # end of the ']'

        return string_rep

    def __repr__(self):
        return str(self)

    def matrix(self, n_qubits=None):
        """
        Convert this fermion operator to csr_matrix under jordan_wigner mapping.

        Args:
            n_qubits (int): The total qubit of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
        """
        from mindquantum.core.operators.utils import count_qubits
        if not self.terms:
            raise ValueError("Cannot convert empty fermion operator to matrix")
        n_qubits_local = count_qubits(self)
        if n_qubits_local == 0 and n_qubits is None:
            raise ValueError("You should specific n_qubits for converting a identity fermion operator.")
        if n_qubits is None:
            n_qubits = n_qubits_local
        _check_int_type("n_qubits", n_qubits)
        if n_qubits < n_qubits_local:
            raise ValueError(
                f"Given n_qubits {n_qubits} is small than qubit of fermion operator, which is {n_qubits_local}.")
        out = 0
        for term, coeff in self.terms.items():
            if isinstance(coeff, PR):
                raise RuntimeError("Cannot convert a parameterized fermion operator to matrix.")
            if not term:
                out += csr_matrix(np.identity(2**n_qubits, dtype=np.complex128)) * coeff
            else:
                tmp = 1
                group = [[]]
                for idx, dag in term:
                    if len(group[-1]) < 4:
                        group[-1].append(idx)
                        group[-1].append(dag)
                    if len(group[-1]) == 4:
                        group.append([])
                for g in group:
                    if g:
                        if len(g) == 4:
                            tmp *= _two_fermion_word(g[0], g[1], g[2], g[3], n_qubits)
                        else:
                            tmp *= _single_fermion_word(g[0], g[1], n_qubits)
                out += tmp * coeff
        return out

    @property
    def imag(self):
        """
        Convert the coeff to its imag part.

        Returns:
            FermionOperator, the imag part of this fermion operator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.imag.compress()
            2.0 [0]
        """
        out = FermionOperator()

        for k, v in self.terms.items():
            out.terms[k] = v.imag
        return out

    @property
    def real(self):
        """
        Convert the coeff to its real part.

        Returns:
            FermionOperator, the real part of this fermion operator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.real.compress()
            1.0 [0] +
            a [0^]
        """
        out = FermionOperator()

        for k, v in self.terms.items():
            out.terms[k] = v.real
        return out

    def normal_ordered(self):
        """Return the normal ordered form of the Fermion Operator.

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
        ordered_op = self.__class__()
        for term, coeff in self.terms.items():
            ordered_op += _normal_ordered_term(term, coeff)
        return ordered_op

    def dumps(self, indent=4):
        '''
        Dump FermionOperator into JSON(JavaScript Object Notation)

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            JSON (str), the JSON strings of FermionOperator

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> print(f.dumps())
            {
                "((0, 0),)": "(1+2j)",
                "((0, 1),)": "{\"a\": 1, \"__class__\": \"ParameterResolver\", \"__module__\": \
                    \"parameterresolver.parameterresolver\", \"no_grad_parameters\": []}",
                "__class__": "FermionOperator",
                "__module__": "operators.fermion_operator"
            }
        '''
        if indent is not None:
            _check_int_type('indent', indent)
        d = self.terms

        # Convert key type from tuple to str
        key_list = list(d.keys())
        for i, k in enumerate(key_list):
            if isinstance(k, tuple):
                key_list[i] = k.__str__()

        # Convert value type from complex/PR into str
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
        Load JSON(JavaScript Object Notation) into FermionOperator

        Args:
            strs (str): The dumped fermion operator string.

        Returns:
            FermionOperator, the FermionOperator load from strings

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> strings == '{"((0, 0),)": "(1+2j)", "((0, 1),)": {"a": 1}, \
                "__class__": "FermionOperator", "__module__": "__main__"}'
            >>> obj = FermionOperator.loads(strings)
            >>> print(obj)
            (1+2j) [0] + a [0^]
        '''
        _check_input_type('strs', str, strs)
        dic = json.loads(strs)
        if '__class__' in dic:
            class_name = dic.pop('__class__')
            if class_name == 'FermionOperator':
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

                f_op = FermionOperator()
                for key, value in terms.items():
                    f_op += class_(key, value)

            else:
                raise TypeError("Require a FermionOperator class, but get {} class".format(class_name))

        else:
            raise ValueError("Expect a '__class__' in strings, but not found")

        return f_op


def _normal_ordered_term(term, coefficient):
    r"""Return the normal ordered term of the FermionOperator with high index
    and creation operator in front.

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
                    new_term = term[:(j - 1)] + term[(j + 1):]
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
    s = []
    for i in term:
        if i[1] == 1:
            s.append('{}^'.format(i[0]))
        else:
            s.append(str(i[0]))
    return ' '.join(s)
