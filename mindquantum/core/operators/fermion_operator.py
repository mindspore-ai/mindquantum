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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""This module is generated the Fermion Operator."""
# pylint: disable=import-error, too-many-public-methods

import copy
import json
import typing
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix, kron

import mindquantum as mq
from mindquantum._math.ops import FermionOperator as FermionOperator_
from mindquantum._math.ops import f_term_value
from mindquantum.core.operators._term_value import TermValue
from mindquantum.core.parameterresolver import ParameterResolver, PRConvertible
from mindquantum.dtype.dtype import str_dtype_map
from mindquantum.mqbackend import EQ_TOLERANCE
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.utils.type_value_check import (
    _check_and_generate_pr_type,
    _check_int_type,
    _require_package,
)


@lru_cache()
def _n_sz(n, dtype):
    if n == 0:
        return csr_matrix(np.array([1]), dtype=dtype)
    tmp = [csr_matrix(np.array([[1, 0], [0, -1]], dtype=dtype)) for _ in range(n)]
    for i in tmp[1:]:
        tmp[0] = kron(tmp[0], i)
    return tmp[0]


@lru_cache()
def _n_identity(n, dtype):
    """N_identity."""
    if n == 0:
        return csr_matrix(np.array([1]), dtype=dtype)
    tmp = [csr_matrix(np.array([[1, 0], [0, 1]], dtype=dtype)) for _ in range(n)]
    for i in tmp[1:]:
        tmp[0] = kron(tmp[0], i)
    return tmp[0]


@lru_cache()
def _single_fermion_word(idx, dag, n_qubits, dtype):
    """Single_fermion_word."""
    matrix = csr_matrix(np.array([[0, 1], [0, 0]], dtype=dtype))
    if dag:
        matrix = csr_matrix(np.array([[0, 0], [1, 0]], dtype=dtype))
    return kron(_n_identity(n_qubits - 1 - idx, dtype), kron(matrix, _n_sz(idx, dtype)))


# pylint: disable=too-many-arguments
@lru_cache()
def _two_fermion_word(idx1, dag1, idx2, dag2, n_qubits, dtype):
    """Two_fermion_word."""
    return _single_fermion_word(idx1, dag1, n_qubits, dtype) * _single_fermion_word(idx2, dag2, n_qubits, dtype)


class FermionOperator(FermionOperator_):
    r"""
    Definition of a Fermion Operator.

    The Fermion Operator such as FermionOperator('9 4^ 3 3^') are used to represent :math:`a_9 a_4^\dagger a_3
    a_3^\dagger`.


    These are the Basic Operators to describe a fermionic system, such as a Molecular system.
    The FermionOperator are follows the anti-commutation relationship.

    Args:
        terms (Union[str, ParameterResolver]): The input term of fermion operator. Default: ``None``.
        coefficient (Union[numbers.Number, str, Dict[str, numbers.Number], ParameterResolver]): The coefficient
            for the corresponding single operators Default: ``1.0``.
        internal (bool): Whether the first argument is internal c++ object of
            FermionOperator or not. Default: ``False``.

    Examples:
        >>> from mindquantum.core.operators import FermionOperator
        >>> a_p_dagger = FermionOperator('1^')
        >>> a_p_dagger
        1 [1^]
        >>> a_q = FermionOperator('0')
        >>> a_q
        1 [0]
        >>> zero = FermionOperator()
        >>> zero
        0
        >>> identity= FermionOperator('')
        >>> identity
        1 []
        >>> para_op = FermionOperator('0 1^', 'x')
        >>> para_op
        -x [1^ 0]
        >>> para_dt = {'x':2}
        >>> op = para_op.subs(para_dt)
        >>> op
        -2 [1^ 0]
    """

    def __init__(
        self,
        terms: typing.Union[str, "FermionOperator"] = None,
        coefficient: PRConvertible = 1.0,
        internal: bool = False,
    ):
        """Initialize a FermionOperator instance."""
        if terms is None:
            FermionOperator_.__init__(self)
        if isinstance(terms, FermionOperator_):
            internal = True
        if internal:
            FermionOperator_.__init__(self, terms)
        else:
            if isinstance(terms, InteractionOperator):
                FermionOperator_.__init__(self, [(i, ParameterResolver(terms[i])) for i in terms])
            else:
                FermionOperator_.__init__(self, terms, ParameterResolver(coefficient))

    def __len__(self) -> int:
        """Return the size of term."""
        return FermionOperator_.size(self)

    def __copy__(self) -> "FermionOperator":
        """Deep copy this FermionOperator."""
        return FermionOperator(FermionOperator_.__copy__(self))

    def __deepcopy__(self, memodict) -> "FermionOperator":
        """Deep copy this FermionOperator."""
        return FermionOperator(FermionOperator_.__copy__(self))

    def __repr__(self) -> str:
        """Return string expression of a FermionOperator."""
        values = []
        terms = []
        max_value_len = 0
        for term, value in self.terms.items():
            values.append(value.expression())
            max_value_len = max(max_value_len, len(values[-1]))
            terms.append("[" + ' '.join(f"{i}{'^' if j else ''}" for i, j in term) + "]")
        for i, j in enumerate(values):
            values[i] = j.rjust(max_value_len)
            if i != len(values) - 1:
                terms[i] += " +"
        if values:
            return "\n".join(f'{v} {t}' for v, t in zip(values, terms))
        return "0"

    def __str__(self) -> str:
        """Return string expression of a FermionOperator."""
        return self.__repr__()

    def __eq__(self, other: typing.Union["FermionOperator", PRConvertible]) -> bool:
        """Check whether two FermionOperator equal or not."""
        if not isinstance(other, FermionOperator_):
            other = ParameterResolver(other, dtype=self.dtype)
            other = FermionOperator("", other)
        return not (self - other).size

    def __add__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Add a number or a FermionOperator."""
        if not isinstance(other, FermionOperator_):
            return FermionOperator(FermionOperator_.__add__(self, FermionOperator("", ParameterResolver(other))))
        return FermionOperator(FermionOperator_.__add__(self, other))

    def __iadd__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Add a number or a FermionOperator."""
        if not isinstance(other, FermionOperator_):
            FermionOperator_.__iadd__(self, FermionOperator("", ParameterResolver(other)))
            return self
        FermionOperator_.__iadd__(self, other)
        return self

    def __radd__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Add a number or a FermionOperator."""
        return self + other

    def __sub__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Sub a number or a FermionOperator."""
        return self + (-1 * other)

    def __isub__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Sub a number or a FermionOperator."""
        self += -1 * other
        return self

    def __rsub__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Sub a number or a FermionOperator."""
        return other + (-1 * self)

    def __neg__(self):
        """Return negative FermionOperator."""
        return 0 - self

    def __mul__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Multiply a number or a FermionOperator."""
        if not isinstance(other, FermionOperator_):
            return FermionOperator(FermionOperator_.__mul__(self, FermionOperator("", ParameterResolver(other))))
        return FermionOperator(FermionOperator_.__mul__(self, other))

    def __imul__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Multiply a number or a FermionOperator."""
        if not isinstance(other, FermionOperator_):
            FermionOperator_.__imul__(self, FermionOperator("", ParameterResolver(other)))
            return self
        FermionOperator_.__imul__(self, other)
        return self

    def __rmul__(self, other: typing.Union["FermionOperator", PRConvertible]) -> "FermionOperator":
        """Multiply a number or a FermionOperator."""
        return self * other

    def __truediv__(self, other: PRConvertible) -> "FermionOperator":
        """Divide a number."""
        if other == 0.0:
            raise ZeroDivisionError("other cannot be zero.")
        return self * (1.0 / other)

    def __itruediv__(self, other: PRConvertible) -> "FermionOperator":
        """Divide a number."""
        if other == 0.0:
            raise ZeroDivisionError("other cannot be zero.")
        self.__imul__(1.0 / other)
        return self

    def __iter__(self) -> typing.Generator["FermionOperator", None, None]:
        """Iterate every single term."""
        for coeff, term in self.split():
            yield term * coeff

    def __pow__(self, frac) -> "FermionOperator":
        """Power of FermionOperator."""
        if not frac:
            return FermionOperator("").astype(self.dtype)
        out = 1 * self
        for _ in range(frac - 1):
            out *= self
        return out

    def __getstate__(self):
        """Get state of parameter resolver."""
        return {'json_str': self.dumps()}

    def __setstate__(self, state):
        """Set state of parameter resolver."""
        a = FermionOperator.loads(state['json_str'])
        self.__init__(a)

    @property
    def constant(self) -> ParameterResolver:
        """
        Return the coefficient of the identity fermion string.

        Returns:
            ParameterResolver, the coefficient of the identity fermion string.
        """
        return ParameterResolver(FermionOperator_.get_coeff(self, []), internal=True)

    @property
    def dtype(self):
        """Get the data type of FermionOperator."""
        return FermionOperator_.dtype(self)

    @property
    def imag(self) -> "FermionOperator":
        """
        Convert the coefficient to its imag part.

        Returns:
            Fermion, the imag part of this FermionOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.imag
            2 [0]
        """
        return FermionOperator(FermionOperator_.imag(self))

    @property
    def is_complex(self) -> bool:
        """Return whether the FermionOperator instance is currently using complex coefficients."""
        return self.dtype in (mq.complex128, mq.complex64)

    @property
    def is_singlet(self) -> bool:
        """
        To verify whether this operator has only one term.

        Returns:
            bool, whether this operator has only one term.
        """
        return FermionOperator_.is_singlet(self)

    @property
    def parameterized(self) -> bool:
        """Check whether this FermionOperator is parameterized."""
        return FermionOperator_.parameterized(self)

    @property
    def real(self) -> "FermionOperator":
        """
        Convert the coefficient to its real part.

        Returns:
            FermionOperator, the real part of this FermionOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.real
            1 [0] +
            a [0^]
        """
        return FermionOperator(FermionOperator_.real(self))

    @property
    def size(self) -> int:
        """Return the number of terms of this FermionOperator."""
        return len(self)

    @property
    def terms(self) -> typing.Dict[typing.Tuple[int, int], ParameterResolver]:
        """Get the terms of a FermionOperator."""
        origin_dict = FermionOperator_.get_terms(self)
        out = {}
        for key, value in origin_dict:
            out_key = []
            for idx, word in key:
                out_key.append((idx, 0 if word == f_term_value.a else 1))
            out[tuple(out_key)] = ParameterResolver(value, internal=True)
        return out

    @staticmethod
    def from_openfermion(of_ops) -> "FermionOperator":
        """
        Convert openfermion fermion operator to mindquantum format.

        Args:
            of_ops (openfermion.FermionOperator): fermion operator from openfermion.

        Returns:
            FermionOperator, fermion operator from mindquantum.
        """
        # pylint: disable=import-outside-toplevel
        try:
            from openfermion import FermionOperator as OFFermionOperator
        except (ImportError, AttributeError):
            _require_package("openfermion", "1.5.0")
        if not isinstance(of_ops, OFFermionOperator):
            raise TypeError(
                "of_ops should be a FermionOperator" f" from openfermion framework, but get type {type(of_ops)}"
            )
        out = FermionOperator()
        for term, v in of_ops.terms.items():
            out += FermionOperator(' '.join([f"{i}{'' if j ==0 else '^'}" for i, j in term]), ParameterResolver(v))
        return out

    @staticmethod
    def loads(strs: str) -> "FermionOperator":
        """
        Load JSON(JavaScript Object Notation) into a FermionOperator.

        Args:
            strs (str): The dumped fermion operator string.

        Returns:
            FermionOperator, the FermionOperator loaded from JSON-formatted strings

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> obj = FermionOperator.loads(f.dumps())
            >>> obj == f
            True
        """
        dic = json.loads(strs)
        out = FermionOperator().astype(str_dtype_map[dic['dtype']])
        for c, t in zip(dic['values'], dic['terms']):
            out += FermionOperator(t, ParameterResolver.loads(c))
        return out

    def astype(self, dtype) -> "FermionOperator":
        """
        Convert to different data type.

        Note:
            Converting a complex type FermionOperator to real type will ignore the image part of coefficient.

        Args:
            dtype (mindquantum.dtype): new data type of fermion operator.

        Returns:
            FermionOperator, new fermion operator with given data type.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> import mindquantum as mq
            >>> f = FermionOperator('0^', 2 + 3j)
            >>> f.dtype
            mindquantum.complex128
            >>> f.astype(mq.float64)
            2 [0^]
        """
        return FermionOperator(FermionOperator_.astype(self, dtype))

    def cast_complex(self) -> "FermionOperator":
        """Cast a FermionOperator into its complex equivalent."""
        new_type = self.dtype
        if new_type == mq.float32:
            new_type = mq.complex64
        elif new_type == mq.float64:
            new_type = mq.complex128
        return self.astype(new_type)

    def compress(self, abs_tol=EQ_TOLERANCE) -> "FermionOperator":
        """
        Eliminate the very small fermion string that close to zero.

        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0. Default: EQ_TOLERANCE.

        Returns:
            FermionOperator, the compressed operator.

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
        out = FermionOperator()
        for k, v in self.terms.items():
            if not (v.is_const() and np.abs(v.const) < abs_tol):
                out += FermionOperator(" ".join(f"{i}{'^' if j else ''}" for i, j in k), v)
        return out

    @constant.setter
    def constant(self, value):
        """Set the coefficient of the Identity term."""
        FermionOperator_.set_coeff(self, [], ParameterResolver(value))

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
        return FermionOperator_.count_qubits(self)

    def dumps(self, indent: int = 4) -> str:
        r"""
        Dump a FermionOperator into JSON(JavaScript Object Notation).

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: ``4``.

        Returns:
            JSON (str), the JSON strings of this FermionOperator

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> len(f.dumps())
            581
        """
        out = {}
        out['dtype'] = str(self.dtype)
        out['terms'] = []
        out['values'] = []
        for k, v in self.terms.items():
            out["values"].append(v.dumps(indent))
            out["terms"].append(" ".join(f"{i}{'^' if j else ''}" for i, j in k))
        return json.dumps(out, indent=indent)

    def get_coeff(self, term) -> ParameterResolver:
        """
        Get coefficient of given term.

        Args:
            term (List[Tuple[int, Union[int, str]]]): the term you want get coefficient.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0^ 1', 1.2)
            >>> f.get_coeff([(1, ''), (0, '^')])
            ParameterResolver(dtype: float64, const: -1.200000)
            >>> f.get_coeff([(1, 0), (0, 1)])
            ParameterResolver(dtype: float64, const: -1.200000)
        """
        return ParameterResolver(FermionOperator_.get_coeff(self, [(i, TermValue[j]) for i, j in term]), internal=True)

    def hermitian(self) -> "FermionOperator":
        """
        Get the hermitian of a FermionOperator.

        Returns:
            FermionOperator, The hermitian of this FermionOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> a = FermionOperator("0^ 1", {"a": 1 + 2j})
            >>> a.hermitian()
            (-1 + 2j)*a [1 0^]
        """
        return FermionOperator(FermionOperator_.hermitian_conjugated(self), internal=True)

    def matrix(self, n_qubits: int = None, pr=None):  # pylint: disable=too-many-branches
        """
        Convert this fermion operator to csr_matrix under jordan_wigner mapping.

        Args:
            n_qubits (int): The total qubit of final matrix. If None, the value will be
                the maximum local qubit number. Default: None.
            pr (ParameterResolver, dict, numpy.ndarray, list, numbers.Number): The parameter
                resolver for parameterized FermionOperator. Default: None.
        """
        if pr is None:
            pr = ParameterResolver()
        pr = _check_and_generate_pr_type(pr, self.params_name)
        np_type = mq.to_np_type(self.dtype)
        ops = self
        if self.parameterized:
            ops = copy.copy(self)
            ops = ops.subs(pr)
        if not self.terms:
            raise ValueError("Cannot convert empty fermion operator to matrix")
        n_qubits_local = ops.count_qubits()
        if n_qubits_local == 0 and n_qubits is None:
            raise ValueError("You should specific n_qubits for converting a identity fermion operator.")
        if n_qubits is None:
            n_qubits = n_qubits_local
        _check_int_type("n_qubits", n_qubits)
        if n_qubits < n_qubits_local:
            raise ValueError(
                f"Given n_qubits {n_qubits} is small than qubit of fermion operator, which is {n_qubits_local}."
            )
        out = 0
        for term, coeff in ops.terms.items():
            coeff = coeff.const
            if not term:
                out += csr_matrix(np.identity(2**n_qubits, dtype=np_type)) * coeff
            else:
                tmp = 1
                group = [[]]
                for idx, dag in term:
                    if len(group[-1]) < 4:
                        group[-1].append(idx)
                        group[-1].append(dag)
                    if len(group[-1]) == 4:
                        group.append([])
                for gate in group:
                    if gate:
                        if len(gate) == 4:
                            tmp *= _two_fermion_word(gate[0], gate[1], gate[2], gate[3], n_qubits, np_type)
                        else:
                            tmp *= _single_fermion_word(gate[0], gate[1], n_qubits, np_type)
                out += tmp * coeff
        return out

    @property
    def params_name(self):
        """Get all parameters of this operator."""
        names = []
        for pr in self.terms.values():
            names.extend([i for i in pr.params_name if i not in names])
        return names

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
        return FermionOperator(FermionOperator_.normal_ordered(self), internal=True)

    def relabel(self, logic_qubits: typing.List[int]) -> "FermionOperator":
        """
        Relabel the qubit according to the given logic qubits order.

        Args:
            logic_qubits (List[int]): The label of logic qubits. For example, if
                logic_qubits is `[2, 0, 1]`, original qubit `0` will label as `2`.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> o = FermionOperator('3^ 2 1 0')
            >>> o
            1 [3^ 2 1 0]
            >>> o.relabel([1, 3, 0, 2])
            -1 [3 2^ 1 0]
        """
        terms = [(tuple((logic_qubits[idx], dag) for idx, dag in key), value) for key, value in self.terms.items()]
        return FermionOperator(terms, internal=True)

    def singlet_coeff(self) -> ParameterResolver:
        """
        Get the coefficient of this operator, if the operator has only one term.

        Returns:
            ParameterResolver, the coefficient of this single string operator.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> ops = FermionOperator("1^ 2", "a")
            >>> print(ops)
            -a [2 1^]
            >>> print(ops.singlet_coeff())
            -a
        """
        return ParameterResolver(FermionOperator_.singlet_coeff(self), internal=True)

    def singlet(self) -> typing.List["FermionOperator"]:
        """
        Split the single string operator into every word.

        Returns:
            List[FermionOperator], The split word of the string.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> ops = FermionOperator("1^ 2", 1)
            >>> print(ops.singlet())
            [1 [2], 1 [1^]]
        """
        return [FermionOperator(i, internal=True) for i in FermionOperator_.singlet(self)]

    def split(self) -> typing.Generator[ParameterResolver, "FermionOperator", None]:
        """
        Split the coefficient and the operator.

        Returns:
            List[List[ParameterResolver, FermionOperator]], the split result.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> a = FermionOperator('0', 'a') + FermionOperator('1^', 1.2)
            >>> for i, j in a.split():
            ...     print(i, j)
            a, 1 [0]
            1.2, 1 [1^]
        """
        for i, j in FermionOperator_.split(self):
            yield ParameterResolver(i, internal=True), FermionOperator(j, internal=True)

    def subs(self, params_value: PRConvertible) -> "FermionOperator":
        """
        Replace the symbolical representation with the corresponding value.

        Args:
            params_value (Union[Dict[str, numbers.Number], ParameterResolver]): the value of variable in coefficient.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> f = FermionOperator('0^', ParameterResolver({'a': 2.0}, 3.0))
            >>> f
            2*a + 3 [0^]
            >>> f.subs({'a': 1.5})
            6 [0^]
        """
        if not isinstance(params_value, ParameterResolver):
            params_value = ParameterResolver(params_value)
        out = copy.copy(self)
        FermionOperator_.subs(out, params_value)
        return out

    def to_openfermion(self):
        """Convert a FermionOperator openfermion format."""
        # pylint: disable=import-outside-toplevel

        try:
            from openfermion import FermionOperator as OFFermionOperator
        except (ImportError, AttributeError):
            _require_package("openfermion", "1.5.0")
        if self.parameterized:
            raise ValueError("Cannot not FermionOperator to OpenFermion format.")

        terms = {}
        for i, j in self.terms.items():
            terms[i] = j.const
        out = OFFermionOperator()
        out.terms = terms
        return out
