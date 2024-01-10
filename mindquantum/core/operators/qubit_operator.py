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
"""This is the module for the Qubit Operator."""
# pylint: disable=import-error

import copy
import json
import typing

import numpy as np
from scipy.sparse import csr_matrix

import mindquantum as mq
from mindquantum._math.ops import QubitOperator as QubitOperator_
from mindquantum.core.operators._term_value import TermValue
from mindquantum.core.parameterresolver import ParameterResolver, PRConvertible
from mindquantum.dtype.dtype import str_dtype_map
from mindquantum.mqbackend import EQ_TOLERANCE
from mindquantum.utils.type_value_check import (
    _check_and_generate_pr_type,
    _check_int_type,
    _require_package,
)


# pylint: disable=too-many-public-methods
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
        term (Union[str, QubitOperator]): The input term of qubit operator. Default: ``None``.
        coefficient (Union[numbers.Number, str, Dict[str, numbers.Number], ParameterResolver]): The
            coefficient of this qubit operator, could be a number or a variable
            represent by a string or a symbol or a parameter resolver. Default: ``1.0``.
        internal (bool): Whether the first argument is internal c++ object of
            QubitOperator or not. Default: ``False``.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator
        >>> ham = ((QubitOperator('X0 Y3', 0.5)
        ...         + 0.6 * QubitOperator('X0 Y3')))
        >>> ham2 = QubitOperator('X0 Y3', 0.5)
        >>> ham2 += 0.6 * QubitOperator('X0 Y3')
        >>> ham2
        1.1 [Y3 X0]
        >>> ham3 = QubitOperator('')
        >>> ham3
        1 []
        >>> ham_para = QubitOperator('X0 Y3', 'x')
        >>> ham_para
        x [Y3 X0]
        >>> ham_para.subs({'x':1.2})
        1.2 [Y3 X0]
    """

    def __init__(
        self,
        terms: typing.Union[str, "QubitOperator"] = None,
        coefficient: PRConvertible = 1.0,
        internal: bool = False,
    ):
        """Initialize a QubitOperator instance."""
        if terms is None:
            QubitOperator_.__init__(self)
        if isinstance(terms, QubitOperator_):
            internal = True
        if internal:
            QubitOperator_.__init__(self, terms)
        else:
            QubitOperator_.__init__(self, terms, ParameterResolver(coefficient))

    def __len__(self) -> int:
        """Return the size of term."""
        return QubitOperator_.size(self)

    def __copy__(self) -> "QubitOperator":
        """Deep copy this QubitOperator."""
        return QubitOperator(QubitOperator_.__copy__(self))

    def __deepcopy__(self, memodict) -> "QubitOperator":
        """Deep copy this QubitOperator."""
        return QubitOperator(QubitOperator_.__copy__(self))

    def __add__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Add a number or a QubitOperator."""
        if not isinstance(other, QubitOperator_):
            return QubitOperator(QubitOperator_.__add__(self, QubitOperator("", ParameterResolver(other))))
        return QubitOperator(QubitOperator_.__add__(self, other))

    def __iadd__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Add a number or a QubitOperator."""
        if not isinstance(other, QubitOperator_):
            QubitOperator_.__iadd__(self, QubitOperator("", ParameterResolver(other)))
            return self
        QubitOperator_.__iadd__(self, other)
        return self

    def __radd__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Add a number or a QubitOperator."""
        return self + other

    def __sub__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Sub a number or a QubitOperator."""
        return self + (-1 * other)

    def __isub__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Sub a number or a QubitOperator."""
        self += -1 * other
        return self

    def __rsub__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Sub a number or a QubitOperator."""
        return other + (-1 * self)

    def __neg__(self):
        """Return negative QubitOperator."""
        return 0 - self

    def __mul__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Multiply a number or a QubitOperator."""
        if not isinstance(other, QubitOperator_):
            return QubitOperator(QubitOperator_.__mul__(self, QubitOperator("", ParameterResolver(other))))
        return QubitOperator(QubitOperator_.__mul__(self, other))

    def __imul__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Multiply a number or a QubitOperator."""
        if not isinstance(other, QubitOperator_):
            QubitOperator_.__imul__(self, QubitOperator("", ParameterResolver(other)))
            return self
        QubitOperator_.__imul__(self, other)
        return self

    def __rmul__(self, other: typing.Union["QubitOperator", PRConvertible]) -> "QubitOperator":
        """Multiply a number or a QubitOperator."""
        return self * other

    def __truediv__(self, other: PRConvertible) -> "QubitOperator":
        """Divide a number."""
        if other == 0.0:
            raise ZeroDivisionError("other cannot be zero.")
        return self * (1.0 / other)

    def __itruediv__(self, other: PRConvertible) -> "QubitOperator":
        """Divide a number."""
        if other == 0.0:
            raise ZeroDivisionError("other cannot be zero.")
        self.__imul__(1.0 / other)
        return self

    def __eq__(self, other: typing.Union["QubitOperator", PRConvertible]) -> bool:
        """Check whether two QubitOperator equal or not."""
        if not isinstance(other, QubitOperator_):
            other = ParameterResolver(other, dtype=self.dtype)
            other = QubitOperator("", other)
        return not (self - other).size

    def __iter__(self) -> typing.Generator["QubitOperator", None, None]:
        """Iterate every single term."""
        for coeff, term in self.split():
            yield term * coeff

    def __pow__(self, frac) -> "QubitOperator":
        """Power of QubitOperator."""
        if not frac:
            return QubitOperator("").astype(self.dtype)
        out = 1 * self
        for _ in range(frac - 1):
            out *= self
        return out

    def __repr__(self) -> str:
        """Return string expression of a QubitOperator."""
        values = []
        terms = []
        max_value_len = 0
        for term, value in self.terms.items():
            values.append(value.expression())
            max_value_len = max(max_value_len, len(values[-1]))
            terms.append("[" + ' '.join(f"{j}{i}" for i, j in term) + "]")
        for i, j in enumerate(values):
            values[i] = j.rjust(max_value_len)
            if i != len(values) - 1:
                terms[i] += " +"
        if values:
            return "\n".join(f'{v} {t}' for v, t in zip(values, terms))
        return "0"

    def __str__(self) -> str:
        """Return string expression of a QubitOperator."""
        return self.__repr__()

    def __getstate__(self):
        """Get state of parameter resolver."""
        return {'json_str': self.dumps()}

    def __setstate__(self, state):
        """Set state of parameter resolver."""
        a = QubitOperator.loads(state['json_str'])
        self.__init__(a)

    @property
    def dtype(self):
        """Get the data type of QubitOperator."""
        return QubitOperator_.dtype(self)

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
        return QubitOperator(QubitOperator_.imag(self), internal=True)

    @property
    def is_complex(self) -> bool:
        """Return whether the QubitOperator instance is currently using complex coefficients."""
        return self.dtype in (mq.complex128, mq.complex64)

    @property
    def is_singlet(self) -> bool:
        """
        To verify whether this operator has only one term.

        Returns:
            bool, whether this operator has only one term.
        """
        return QubitOperator_.is_singlet(self)

    @property
    def parameterized(self) -> bool:
        """Check whether this QubitOperator is parameterized."""
        return QubitOperator_.parameterized(self)

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
        return QubitOperator(QubitOperator_.real(self), internal=True)

    @property
    def size(self) -> int:
        """Return the number of terms of this QubitOperator."""
        return len(self)

    @property
    def terms(self) -> typing.Dict[typing.Tuple[int, str], ParameterResolver]:
        """Get the terms of a QubitOperator."""
        origin_dict = QubitOperator_.get_terms(self)
        out = {}
        for key, value in origin_dict:
            out_key = []
            for idx, t in key:
                out_key.append((idx, str(t)))
            out[tuple(out_key)] = ParameterResolver(value, internal=True)
        return out

    @staticmethod
    def from_openfermion(of_ops) -> "QubitOperator":
        """
        Convert qubit operator from openfermion to mindquantum format.

        Args:
            of_ops (openfermion.QubitOperator): Qubit operator from openfermion.

        Returns:
            QubitOperator, qubit operator from mindquantum.
        """
        # pylint: disable=import-outside-toplevel
        try:
            from openfermion import QubitOperator as OFQubitOperator
        except (ImportError, AttributeError):
            _require_package("openfermion", "1.5.0")
        if not isinstance(of_ops, OFQubitOperator):
            raise TypeError(
                "of_ops should be a QubitOperator" f" from openfermion framework, but get type {type(of_ops)}"
            )
        out = QubitOperator()
        for term, v in of_ops.terms.items():
            out += QubitOperator(' '.join([f"{j}{i}" for i, j in term]), ParameterResolver(v))
        return out

    @staticmethod
    def loads(strs: str) -> "QubitOperator":
        """
        Load JSON(JavaScript Object Notation) into a QubitOperator.

        Args:
            strs (str): The dumped fermion operator string.

        Returns:
            QubitOperator, the QubitOperator loaded from JSON-formatted strings.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('0', 1 + 2j) + QubitOperator('0^', 'a')
            >>> obj = QubitOperator.loads(f.dumps())
            >>> obj == f
            True
        """
        dic = json.loads(strs)
        out = QubitOperator().astype(str_dtype_map[dic['dtype']])
        for c, t in zip(dic['values'], dic['terms']):
            out += QubitOperator(t, ParameterResolver.loads(c))
        return out

    def astype(self, dtype) -> "QubitOperator":
        """
        Convert to different data type.

        Note:
            Converting a complex type QubitOperator to real type will ignore the image part of coefficient.

        Args:
            dtype (mindquantum.dtype): new data type of fermion operator.

        Returns:
            QubitOperator, new fermion operator with given data type.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> import mindquantum as mq
            >>> f = QubitOperator('X0', 2 + 3j)
            >>> f.dtype
            mindquantum.complex128
            >>> f.astype(mq.float64)
            2 [X0]
        """
        return QubitOperator(QubitOperator_.astype(self, dtype))

    def cast_complex(self) -> "QubitOperator":
        """Cast a QubitOperator into its complex equivalent."""
        new_type = self.dtype
        if new_type == mq.float32:
            new_type = mq.complex64
        elif new_type == mq.float64:
            new_type = mq.complex128
        return self.astype(new_type)

    def compress(self, abs_tol=EQ_TOLERANCE) -> "QubitOperator":
        """
        Eliminate the very small pauli string that close to zero.

        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0. Default: EQ_TOLERANCE.

        Returns:
            QubitOperator, the compressed operator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ham_compress = QubitOperator('X0 Y1', 0.5) + QubitOperator('Z1 X3', 1e-7)
            >>> ham_compress
            1/2 [Y1 X0] +
            1/10000000 [X3 Z1]
            >>> ham_compress.compress(1e-6)
            1/2 [Y1 X0]
            >>> ham_para_compress =  QubitOperator('X0 Y1', 0.5) + QubitOperator('Z5', 'X')
            >>> ham_para_compress
            1/2 [Y1 X0] +
            X [Z5]
            >>> ham_para_compress.compress(1e-7)
            1/2 [Y1 X0] +
            X [Z5]
        """
        out = QubitOperator()
        for k, v in self.terms.items():
            if not (v.is_const() and np.abs(v.const) < abs_tol):
                out += QubitOperator(" ".join(f"{j}{i}" for i, j in k), v)
        return out

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
        return sum(len(t) for t in self.terms)

    def count_qubits(self) -> int:
        """
        Calculate the number of qubits on which operator acts before removing the unused qubit.

        Returns:
            int, the qubits number before remove unused qubit.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> a = QubitOperator("Z0 Y3")
            >>> a.count_qubits()
            4
        """
        return QubitOperator_.count_qubits(self)

    def dumps(self, indent: int = 4) -> str:
        r"""
        Dump a QubitOperator into JSON(JavaScript Object Notation).

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            JSON (str), the JSON strings of this QubitOperator

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('0', 1 + 2j) + QubitOperator('0^', 'a')
            >>> len(f.dumps())
            581
        """
        out = {}
        out['dtype'] = str(self.dtype)
        out['terms'] = []
        out['values'] = []
        for k, v in self.terms.items():
            out["values"].append(v.dumps(indent))
            out["terms"].append(" ".join(f"{str(j)}{i}" for i, j in k))
        return json.dumps(out, indent=indent)

    def get_coeff(self, term) -> ParameterResolver:
        """
        Get coefficient of given term.

        Args:
            term (List[Tuple[int, Union[int, str]]]): the term you want get coefficient.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> q = QubitOperator('X0 Y1', 1.2)
            >>> q.get_coeff([(1, 'Y'), (0, 'X')])
            ParameterResolver(dtype: float64, const: 1.200000)
        """
        return ParameterResolver(QubitOperator_.get_coeff(self, [(i, TermValue[j]) for i, j in term]), internal=True)

    def hermitian(self) -> "QubitOperator":
        """
        Get the hermitian of a QubitOperator.

        Returns:
            QubitOperator, the hermitian of this QubitOperator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> a = QubitOperator("X0 Y1", {"a": 1 + 2j})
            >>> a.hermitian()
            (-1 + 2j)*a [1 0^]
        """
        return QubitOperator(QubitOperator_.hermitian_conjugated(self), internal=True)

    def matrix(self, n_qubits: int = None, pr=None):
        """
        Convert this qubit operator to csr_matrix.

        Args:
            n_qubits (int): The total qubits of final matrix. If ``None``, the value will be
                the maximum local qubit number. Default: ``None``.
            pr (ParameterResolver, dict, numpy.ndarray, list, numbers.Number): The parameter
                resolver for parameterized QubitOperator. Default: None.
        """
        if pr is None:
            pr = ParameterResolver()
        pr = _check_and_generate_pr_type(pr, self.params_name)
        ops = self
        if self.parameterized:
            ops = copy.copy(self)
            ops = ops.subs(pr)
        if n_qubits is None:
            n_qubits = -1
        _check_int_type('n_qubits', n_qubits)
        csr = QubitOperator_.sparsing(ops, n_qubits)
        data = np.array(csr.data, copy=False)
        indptr = np.array(csr.get_indptr(), copy=False)
        indices = np.array(csr.get_indices(), copy=False)
        return csr_matrix((data, indices, indptr), (csr.n_row, csr.n_col))

    @property
    def params_name(self):
        """Get all parameters of this operator."""
        names = []
        for pr in self.terms.values():
            names.extend([i for i in pr.params_name if i not in names])
        return names

    def relabel(self, logic_qubits: typing.List[int]) -> "QubitOperator":
        """
        Relabel the qubit according to the given logic qubits order.

        Args:
            logic_qubits (List[int]): The label of logic qubits. For example, if
                logic_qubits is `[2, 0, 1]`, original qubit `0` will label as `2`.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> o = QubitOperator('Z0 Y1 X2 Z3')
            >>> o
            1 [Z0 Y1 X2 Z3]
            >>> o.relabel([1, 3, 0, 2])
            1 [X0 Z1 Z2 Y3]
        """
        terms = [(tuple((logic_qubits[idx], dag) for idx, dag in key), value) for key, value in self.terms.items()]
        return QubitOperator(terms, internal=True)

    def singlet(self) -> typing.List["QubitOperator"]:
        """
        Split the single string operator into every word.

        Returns:
            List[QubitOperator], The split word of the string.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator("1^ 2", 1)
            >>> print(ops.singlet())
            [1 [2], 1 [1^]]
        """
        return [QubitOperator(i, internal=True) for i in QubitOperator_.singlet(self)]

    def singlet_coeff(self) -> ParameterResolver:
        """
        Get the coefficient of this operator, if the operator has only one term.

        Returns:
            ParameterResolver, the coefficient of this single string operator.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator("X0 Y1", "a")
            >>> print(ops)
            -a [2 1^]
            >>> print(ops.singlet_coeff())
            -a
        """
        return ParameterResolver(QubitOperator_.singlet_coeff(self), internal=True)

    def split(self):
        """
        Split the coefficient and the operator.

        Returns:
            List[List[ParameterResolver, QubitOperator]], the split result.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> a = QubitOperator('X0', 'a') + QubitOperator('Z1', 1.2)
            >>> for i, j in a.split():
            ...     print(f"{i}, {j}")
            a, 1 [X0]
            1.2, 1 [Z1]
        """
        for i, j in QubitOperator_.split(self):
            yield ParameterResolver(i, internal=True), QubitOperator(j, internal=True)

    def subs(self, params_value: PRConvertible) -> "QubitOperator":
        """
        Replace the symbolical representation with the corresponding value.

        Args:
            params_value (Union[Dict[str, numbers.Number], ParameterResolver]): the value of variable in coefficient.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> q = QubitOperator('X0', ParameterResolver({'a': 2.0}, 3.0))
            >>> q
            2*a + 3 [X0]
            >>> q.subs({'a': 1.5})
            6 [X0]
        """
        if not isinstance(params_value, ParameterResolver):
            params_value = ParameterResolver(params_value)
        out = copy.copy(self)
        QubitOperator_.subs(out, params_value)
        return out

    def to_openfermion(self):
        """Convert qubit operator to openfermion format."""
        # pylint: disable=import-outside-toplevel

        try:
            from openfermion import QubitOperator as OFQubitOperator
        except (ImportError, AttributeError):
            _require_package("openfermion", "1.5.0")
        if self.parameterized:
            raise ValueError("Cannot not QubitOperator to OpenFermion format.")

        terms = {}
        for i, j in self.terms.items():
            terms[i] = j.const
        out = OFQubitOperator()
        out.terms = terms
        return out
