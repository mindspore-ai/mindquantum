# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import json
import numbers
import typing

from mindquantum._math.ops import FermionOperator as FermionOperator_
from mindquantum._math.ops import f_term_value
from mindquantum.core.tensor import dtype as mqtype
from mindquantum.core.tensor.parameterresolver import ParameterResolver, PRConvertible
from mindquantum.core.tensor.term_value import TermValue
from mindquantum.utils.type_value_check import _require_package


class FermionOperator(FermionOperator_):
    r"""
    Definition of a Fermion Operator.

    The Fermion Operator such as FermionOperator('9 4^ 3 3^') are used to represent :math:`a_9 a_4^\dagger a_3
    a_3^\dagger`.


    These are the Basic Operators to describe a fermionic system, such as a Molecular system.
    The FermionOperator are follows the anti-commutation relationship.

    Args:
        terms (str): The input term of fermion operator. Default: None.
        coefficient (Union[numbers.Number, str, Dict[str, numbers.Number], ParameterResolver]): The coefficient
            for the corresponding single operators Default: 1.0.

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
            FermionOperator_.__init__(self, terms, ParameterResolver(coefficient))

    def __len__(self) -> int:
        """Return the size of term."""
        return FermionOperator_.size(self)

    def __copy__(self) -> "FermionOperator":
        """Deep copy this FermionOperator."""
        return FermionOperator(FermionOperator_.__copy__(self))

    def __deepcopy__(self) -> "FermionOperator":
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
        return self * (1.0 / other)

    def __itruediv__(self, other: PRConvertible) -> "FermionOperator":
        """Divide a number."""
        self.__imul__(1.0 / other)
        return self

    def __iter__(self) -> typing.Generator["FermionOperator", None, None]:
        """Iterate every single term."""
        for coeff, term in self.split():
            yield term * coeff

    def astype(self, dtype) -> "FermionOperator":
        """Convert to different type."""
        return FermionOperator(FermionOperator_.astype(self, dtype))

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
    def terms(self) -> typing.Dict[typing.Tuple[int, int], ParameterResolver]:
        """Get the terms of a FermionOperator."""
        origin_dict = FermionOperator_.get_terms(self)
        out = {}
        for key, value in origin_dict:
            out_key = []
            for idx, t in key:
                out_key.append((idx, 0 if t == f_term_value.a else 1))
            out[tuple(out_key)] = ParameterResolver(value, internal=True)
        return out

    @property
    def constant(self) -> ParameterResolver:
        """Return the value of the constant term."""
        return ParameterResolver(FermionOperator_.get_coeff(self, []), internal=True)

    @constant.setter
    def constant(self, value):
        """Set the coefficient of the Identity term."""
        FermionOperator_.set_coeff(self, [], ParameterResolver(value))

    def cast_complex(self) -> "FermionOperator":
        """Cast a FermionOperator into its complex equivalent."""
        new_type = self.dtype
        if new_type == mqtype.float32:
            new_type = mqtype.complex64
        elif new_type == mqtype.float64:
            new_type = mqtype.complex128
        return self.astype(new_type)

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

    def get_coeff(self, term) -> ParameterResolver:
        """Get coefficient of given term."""
        return ParameterResolver(FermionOperator_.get_coeff(self, [(i, TermValue[j]) for i, j in term]), internal=True)

    def hermitian(self) -> "FermionOperator":
        """
        Get the hermitian of a FermionOperator.

        Returns:
            The hermitian of this FermionOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> a = FermionOperator("0^ 1", {"a": 1 + 2j})
            >>> a.hermitian()
            (-1 + 2j)*a [1 0^]
        """
        return FermionOperator(FermionOperator_.hermitian_conjugated(self), internal=True)

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
            a 1 [0]
            1.2 1 [1^]
        """
        for i, j in FermionOperator_.split(self):
            yield ParameterResolver(i, internal=True), FermionOperator(j, internal=True)

    def parameterized(self) -> bool:
        """Check whether this FermionOperator is parameterized."""
        return FermionOperator_.parameterized(self)

    def to_openfermion(self):
        """Convert a FermionOperator openfermion format."""
        # pylint: disable=import-outside-toplevel

        try:
            from openfermion import FermionOperator as o_fo
        except (ImportError, AttributeError):
            _require_package("openfermion", "1.5.0")
        if self.parameterized():
            raise ValueError("Cannot not FermionOperator to OpenFermion format.")

        terms = {}
        for i, j in self.terms.items():
            terms[i] = j.const
        out = o_fo()
        out.terms = terms
        return out

    @staticmethod
    def from_openfermion(of_ops) -> "FermionOperator":
        """
        Convert openfermion fermion operator to mindquantum format.

        Args:
            of_ops: fermion operator from openfermion.

        Returns:
            FermionOperator, terms operator from mindquantum.
        """
        # pylint: disable=import-outside-toplevel
        try:
            from openfermion import FermionOperator as f_of
        except (ImportError, AttributeError):
            _require_package("openfermion", "1.5.0")
        if not isinstance(of_ops, f_of):
            raise TypeError(
                "of_ops should be a FermionOperator" f" from openfermion framework, but get type {type(of_ops)}"
            )
        out = FermionOperator()
        for term, v in of_ops.terms.items():
            out += FermionOperator(' '.join([f"{i}{'' if j ==0 else '^'}" for i, j in term]), ParameterResolver(v))
        return out

    @property
    def size(self) -> int:
        """Return the number of terms of this FermionOperator."""
        return len(self)

    @property
    def is_singlet(self) -> bool:
        """
        To verify whether this operator has only one term.

        Returns:
            bool, whether this operator has only one term.
        """
        return FermionOperator_.is_singlet(self)

    @property
    def is_complexs(self) -> bool:
        """Return whether the FermionOperator instance is currently using complex coefficients."""
        return self.dtype in (mqtype.complex128, mqtype.complex64)

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
            >>> print(ops)
            -a [2 1^]
            >>> print(ops.singlet_coeff())
            -a
        """
        return ParameterResolver(FermionOperator_.singlet_coeff(self), internal=True)

    def singlet(self) -> typing.List["FermionOperator"]:
        """
        Split the single string operator into every word.

        Raises:
            RuntimeError: if the size of terms is not equal to 1.

        Returns:
            List[FermionOperator]: The split word of the string.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> ops = FermionOperator("1^ 2", 1)
            >>> print(ops.singlet())
            [1 [2], 1 [1^]]
        """
        return [FermionOperator(i, internal=True) for i in FermionOperator_.singlet(self)]

    def subs(self, params_value: typing.Union[typing.Dict[str, numbers.Number], ParameterResolver]):
        """Replace the symbolical representation with the corresponding value."""
        if not isinstance(params_value, ParameterResolver):
            params_value = ParameterResolver(params_value)
        out = self.__copy__()
        FermionOperator_.subs(out, params_value)
        return out

    def dumps(self, indent: int = 4) -> str:
        r"""
        Dump a FermionOperator into JSON(JavaScript Object Notation).

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

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
        out = FermionOperator().astype(mqtype.str_dtype_map[dic['dtype']])
        for c, t in zip(dic['values'], dic['terms']):
            out += FermionOperator(t, ParameterResolver.loads(c))
        return out

    def __pow__(self, frac) -> "FermionOperator":
        """Power of FermionOperator."""
        out = frac
        for i in range(frac - 1):
            out *= self
        return self


if __name__ == "__main__":
    a = FermionOperator('9 4^ 3 3^')
    b = FermionOperator('2')
    c = a * b
    d = b * a
