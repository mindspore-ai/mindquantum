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
"""ParameterResolver module."""

# pylint: disable=too-many-lines,import-error,too-many-branches,too-many-public-methods
import json
import numbers
import typing

import numpy as np

import mindquantum as mq
from mindquantum._math.pr import ParameterResolver as ParameterResolver_
from mindquantum._math.tensor import from_numpy
from mindquantum.dtype.dtype import mq_complex_number_type, str_dtype_map
from mindquantum.utils.string_utils import join_without_empty, string_expression
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_mq_type,
)

PRConvertible = typing.Union[numbers.Number, str, typing.Dict[str, numbers.Number], "ParameterResolver"]


class ParameterResolver(ParameterResolver_):
    """
    A ParameterResolver can set the parameter of parameterized quantum gate or parameterized quantum circuit.

    Args:
        data (Union[dict, numbers.Number, str, ParameterResolver]): initial parameter names and
            its values. If data is a dict, the key will be the parameter name
            and the value will be the parameter value. If data is a number, this
            number will be the constant value of this parameter resolver. If data
            is a string, then this string will be the only parameter with coefficient
            be 1. Default: ``None``.
        const (number.Number): the constant part of this parameter resolver.
            Default: ``None``.
        dtype (mindquantum.dtype): the data type of this parameter resolver. Default: ``None``.
        internal (bool): whether the first argument is the c++ object of parameter resolver. Default: ``False``.

    Examples:
        >>> from mindquantum.core.parameterresolver import ParameterResolver
        >>> pr = ParameterResolver({'a': 0.3})
        >>> pr['b'] = 0.5
        >>> pr.no_grad_part('a')
        ParameterResolver(dtype: float64,
        data: [
                a: 0.300000,
                b: 0.500000
        ],
        const: 0.000000,
        no grad parameters: {a, }
        )
        >>> pr *= 2
        >>> pr
        ParameterResolver(dtype: float64,
        data: [
                a: 0.600000,
                b: 1.000000
        ],
        const: 0.000000,
        no grad parameters: {a, }
        )
        >>> pr.expression()
        '0.6*a + b'
        >>> pr.const = 0.5
        >>> pr.expression()
        '0.6*a + b + 1/2'
        >>> pr.no_grad_parameters
        ['a']
        >>> ParameterResolver(3)
        ParameterResolver(dtype: float64, const: 3.000000)
        >>> ParameterResolver('a')
        ParameterResolver(dtype: float64,
        data: [
                a: 1.000000
        ],
        const: 0.000000
        )
    """

    def __init__(self, data=None, const=None, dtype=None, internal=False):
        """Initialize a ParameterResolver object."""
        if dtype is not None:
            _check_mq_type(dtype)
        if isinstance(data, ParameterResolver):
            internal = True
        if internal:
            if dtype is not None:
                ParameterResolver_.__init__(self, data.astype(dtype))
            else:
                ParameterResolver_.__init__(self, data)
        else:
            if isinstance(data, str):
                if dtype is None:
                    dtype = mq.float64
                    if const is not None:
                        if isinstance(const, numbers.Number) and not isinstance(const, numbers.Real):
                            dtype = mq.complex128
                if const is None:
                    const = from_numpy(np.array([0.0], dtype=mq.to_np_type(dtype)))
                else:
                    const = from_numpy(np.array([const], dtype=mq.to_np_type(dtype)))
                ParameterResolver_.__init__(self, data, const, dtype)  # PR('a'[, 1.0, mq.float64])
            elif isinstance(data, dict):
                if dtype is None:
                    dtype = mq.float64
                    for v in data.values():
                        if isinstance(v, numbers.Number) and not isinstance(v, numbers.Real):
                            dtype = mq.complex128
                            break
                    if const is not None:
                        if isinstance(const, numbers.Number) and not isinstance(const, numbers.Real):
                            dtype = mq.complex128
                if const is None:
                    const = from_numpy(np.array([0.0], dtype=mq.to_np_type(dtype)))
                else:
                    const = from_numpy(np.array([const], dtype=mq.to_np_type(dtype)))
                # PR({'a': 1.0}[, 2.0, mq.float64])
                ParameterResolver_.__init__(
                    self,
                    {i: from_numpy(np.array([j], dtype=mq.to_np_type(dtype))) for i, j in data.items()},
                    const,
                    dtype,
                )
            elif isinstance(data, numbers.Number):
                if dtype is None:
                    dtype = mq.float64
                    if isinstance(data, numbers.Number) and not isinstance(data, numbers.Real):
                        dtype = mq.complex128
                ParameterResolver_.__init__(
                    self, from_numpy(np.array([data], dtype=mq.to_np_type(dtype)))
                )  # PR(1.0[, mq.float64])
            elif data is None:
                ParameterResolver_.__init__(self)
            else:
                raise ValueError(
                    "data requires a number or a ParameterResolver or a dict " f"or a string, but get {type(data)}"
                )

    def __str__(self) -> str:
        """Return the string expression of this parameter resolver."""
        return self.expression()

    def __repr__(self) -> str:
        """Return the repr of this parameter resolver."""
        return ParameterResolver_.__str__(self)

    def __len__(self) -> int:
        """
        Get the number of parameters in this parameter resolver.

        Please note that the parameter with 0 coefficient is also considered.

        Returns:
            int, the number of all parameters.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 0, 'b': 1})
            >>> a.expression()
            'b'
            >>> len(a)
            2
        """
        return ParameterResolver_.__len__(self)

    def __contains__(self, key: str) -> bool:
        """
        Check whether the given key is in this parameter resolver or not.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> 'c' in pr
            False
        """
        return ParameterResolver_.__contains__(self, key)

    def __copy__(self) -> "ParameterResolver":
        """
        Copy a new parameter resolver.

        Returns:
            ParameterResolver, the new parameter resolver you copied.

        Examples:
            >>> import copy
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 2}, 3)
            >>> b = copy.copy(a)
            >>> c = a
            >>> a['a'] = 4
            >>> a.expression()
            '4*a + 3'
            >>> b.expression()
            '2*a + 3'
            >>> c.expression()
            '4*a + 3'
        """
        return ParameterResolver(self, internal=True)

    def __deepcopy__(self, memo) -> "ParameterResolver":
        """Deep copy operator."""
        return ParameterResolver(self, internal=True)

    def __eq__(self, other: PRConvertible) -> bool:
        """
        To check whether two parameter resolvers are equal.

        Args:
            other (Union[numbers.Number, str, ParameterResolver]): The parameter resolver
                or number you want to compare. If a number or string is given, this number will
                convert to a parameter resolver.

        Returns:
            bool, whether two parameter resolvers are equal.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> PR(3) == 3
            True
            >>> PR({'a': 2}, 3) == PR({'a': 2}) + 3
            True
        """
        return not bool(self - other)

    def __bool__(self) -> bool:
        """
        Check whether this parameter resolver has non zero constant or parameter with non zero coefficient.

        Returns:
            bool, False if this parameter resolver represent zero and True if not.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR(0)
            >>> bool(pr)
            False
        """
        return ParameterResolver_.is_not_zero(self)

    def __setitem__(self, key: str, value: numbers.Number):
        """
        Set the value of parameter in this parameter resolver.

        You can set multiple values of multiple parameters with given iterable keys and values.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR(0)
            >>> pr['a'] = 2.5
            >>> pr.expression()
            '5/2*a'
        """
        ParameterResolver_.set_item(self, key, from_numpy(np.array([value])))

    def __getitem__(self, key: str) -> numbers.Number:
        """
        Get the parameter value from this parameter resolver.

        Returns:
            numbers.Number, the parameter value.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> pr['a']
            1.0
        """
        return np.array(ParameterResolver_.get_item(self, key))[0]

    def __iter__(self) -> typing.Generator[str, None, None]:
        """
        Yield the parameter name.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> list(pr)
            ['a', 'b']
        """
        yield from self.keys()

    def __float__(self) -> float:
        """Convert the constant part to float. Raise error if it's not constant."""
        if not self.is_const():
            raise ValueError("parameter resolver is not constant, cannot convert to float.")
        return np.float64(self.const)

    def __iadd__(self, other: PRConvertible) -> "ParameterResolver":
        """
        Inplace add a number or parameter resolver.

        Args (Union[numbers.Number, ParameterResolver]): the number or parameter
            resolver you want add.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 3})
            >>> pr += 4
            >>> pr.expression()
            '3*a + 4'
            >>> pr += PR({'b': 1.5})
            >>> pr
            ParameterResolver(dtype: float64,
            data: [
                    a: 3.000000,
                    b: 1.500000
            ],
            const: 4.000000
            )
        """
        if isinstance(other, ParameterResolver_):
            ParameterResolver_.__iadd__(self, other)
        else:
            ParameterResolver_.__iadd__(self, ParameterResolver(other))
        return self

    def __add__(self, other: PRConvertible) -> "ParameterResolver":
        """
        Add a number or parameter resolver.

        Args:
            other (Union[numbers.Number, ParameterResolver])

        Returns:
            ParameterResolver, the result of add.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 3})
            >>> (pr + 1).expression()
            '3*a + 1'
            >>> (pr + pr).expression()
            '6*a'
        """
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(ParameterResolver_.__add__(self, other), internal=True)
        if not isinstance(other, (numbers.Number, str, dict)):
            return other.__radd__(self)
        return ParameterResolver(ParameterResolver_.__add__(self, ParameterResolver(other)), internal=True)

    def __radd__(self, other: PRConvertible) -> "ParameterResolver":
        """Add a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(other + self, internal=True)
        return ParameterResolver(ParameterResolver(other) + self, internal=True)

    def __isub__(self, other: PRConvertible):
        """Self subtract a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            ParameterResolver_.__isub__(self, other)
        else:
            ParameterResolver_.__isub__(self, ParameterResolver(other))
        return self

    def __sub__(self, other: PRConvertible):
        """Subtract a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(ParameterResolver_.__sub__(self, other), internal=True)
        if not isinstance(other, (numbers.Number, str, dict)):
            return other.__rsub__(self)
        return ParameterResolver(ParameterResolver_.__sub__(self, ParameterResolver(other)), internal=True)

    def __rsub__(self, other: PRConvertible):
        """Self subtract a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return other - self
        return ParameterResolver(other) - self

    def __imul__(self, other: PRConvertible):
        """Self multiply a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            ParameterResolver_.__imul__(self, other)
        else:
            ParameterResolver_.__imul__(self, ParameterResolver(other))
        return self

    def __mul__(self, other: PRConvertible):
        """Multiply a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(ParameterResolver_.__mul__(self, other), internal=True)
        if not isinstance(other, (numbers.Number, str, dict)):
            return other.__rmul__(self)
        return ParameterResolver(ParameterResolver_.__mul__(self, ParameterResolver(other)), internal=True)

    def __rmul__(self, other: PRConvertible):
        """Multiply a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return other * self
        return ParameterResolver(other) * self

    def __neg__(self):
        """Return the negative of this parameter resolver."""
        return 0 - self

    def __itruediv__(self, other: PRConvertible):
        """Divide a number inplace."""
        if isinstance(other, ParameterResolver_):
            ParameterResolver_.__itruediv__(self, other)
        else:
            ParameterResolver_.__itruediv__(self, ParameterResolver(other))
        return self

    def __truediv__(self, other: PRConvertible):
        """Divide a number."""
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(ParameterResolver_.__truediv__(self, other), internal=True)
        return ParameterResolver(ParameterResolver_.__truediv__(self, ParameterResolver(other)), internal=True)

    def __getstate__(self):
        """Get state of parameter resolver."""
        return {'json_str': self.dumps()}

    def __setstate__(self, state):
        """Set state of parameter resolver."""
        a = ParameterResolver.loads(state['json_str'])
        self.__init__(a)

    @property
    def ansatz_parameters(self) -> typing.List[str]:
        """
        Get parameters that is ansatz parameters.

        Returns:
            set, the set of ansatz parameters.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.ansatz_parameters
            ['a', 'b']
        """
        return [i for i in self.params_name if i not in self.encoder_parameters]

    @property
    def const(self) -> numbers.Number:
        """
        Get the constant part of this parameter resolver.

        Returns:
            numbers.Number, the constant part of this parameter resolver.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1}, 2.5)
            >>> pr.const
            2.5
        """
        return np.array(ParameterResolver_.get_const(self))[0]

    @property
    def dtype(self):
        """Get ParameterResolver data type."""
        return ParameterResolver_.dtype(self)

    @property
    def encoder_parameters(self) -> typing.List[str]:
        """
        Get parameters that is encoder parameters.

        Returns:
            set, the set of encoder parameters.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.as_encoder()
            >>> a.encoder_parameters
            ['b', 'a']
        """
        return list(ParameterResolver_.get_encoder_parameters(self))

    @property
    def imag(self) -> "ParameterResolver":
        """
        Get the imaginary part of every parameter value.

        Returns:
            ParameterResolver, image part parameter value.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR('a', 3) + 1j * PR('a', 4)
            >>> pr
            ParameterResolver(dtype: complex128,
            data: [
                    a: (1.000000, 1.000000)
            ],
            const: (3.000000, 4.000000)
            )
            >>> pr.imag
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000
            ],
            const: 4.000000
            )
        """
        return ParameterResolver(ParameterResolver_.imag(self), internal=True)

    @property
    def is_complex(self) -> bool:
        """Return whether the ParameterResolver instance is currently using complex coefficients."""
        return self.dtype in mq_complex_number_type

    @property
    def no_grad_parameters(self) -> typing.List[str]:
        """
        Get parameters that do not require grad.

        Returns:
            set, the set of parameters that do not require grad.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.no_grad()
            >>> a.no_grad_parameters
            ['b', 'a']
        """
        return list(ParameterResolver_.get_grad_parameters(self))

    @property
    def params_name(self) -> typing.List[str]:
        """
        Get the parameters name.

        Returns:
            list, a list of parameters name.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.params_name
            ['a', 'b']
        """
        return list(self.keys())

    @property
    def params_value(self) -> typing.List[numbers.Number]:
        """
        Get the parameters value.

        Returns:
            list, a list of parameters value.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.params_value
            [1.0, 2.0]
        """
        return list(self.values())

    @property
    def real(self) -> "ParameterResolver":
        """
        Get the real part of every parameter value.

        Returns:
            ParameterResolver, real part parameter value.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR('a', 3) + 1j * PR('a', 4)
            >>> pr
            ParameterResolver(dtype: complex128,
            data: [
                    a: (1.000000, 1.000000)
            ],
            const: (3.000000, 4.000000)
            )
            >>> pr.real
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000
            ],
            const: 3.000000
            )
        """
        return ParameterResolver(ParameterResolver_.real(self), internal=True)

    @property
    def requires_grad_parameters(self) -> typing.List[str]:
        """
        Get parameters that requires grad.

        Returns:
            set, the set of parameters that requires grad.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.requires_grad_parameters
            ['a', 'b']
        """
        return [i for i in self.params_name if i not in self.no_grad_parameters]

    @staticmethod
    def loads(strs: str) -> "ParameterResolver":
        r"""
        Load JSON(JavaScript Object Notation) into FermionOperator.

        Args:
            strs (str): The dumped parameter resolver string.

        Returns:
            FermionOperator, the FermionOperator load from strings

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> ori = ParameterResolver({'a': 1, 'b': 2, 'c': 3, 'd': 4})
            >>> ori.no_grad_part('a', 'b')
            >>> string = ori.dumps()
            >>> obj = ParameterResolver.loads(string)
            >>> print(obj)
            a + 2*b + 3*c + 4*d
            >>> print('requires_grad_parameters is:', obj.requires_grad_parameters)
            requires_grad_parameters is: ['c', 'd']
            >>> print('no_grad_parameters is :', obj.no_grad_parameters)
            no_grad_parameters is : ['b', 'a']
        """
        _check_input_type('strs', str, strs)
        dic = json.loads(strs)
        if 'dtype' not in dic:
            raise ValueError("Invalid string. Cannot convert it to ParameterResolver, no key dtype")
        dtype = str_dtype_map[dic['dtype']]
        if dtype in mq_complex_number_type:
            const = dic['const'][0] + 1j * dic['const'][1]
            data = {i: j[0] + j[1] * 1j for i, j in dic['pr_data'].items()}
        else:
            const = dic['const'][0]
            data = {i: j[0] for i, j in dic['pr_data'].items()}
        out = ParameterResolver(data, const, dtype)
        out.no_grad_part(*dic['no_grad_parameters'])
        out.encoder_part(*dic['encoder_parameters'])
        return out

    def ansatz_part(self, *names):
        """
        Set which part is ansatz parameters.

        Args:
            names (tuple[str]): Parameters that will be serve as ansatz.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.as_encoder()
            >>> pr.ansatz_part('a')
            >>> pr.ansatz_parameters
            ['a']
        """
        for name in names:
            _check_input_type('name', str, name)
        ParameterResolver_.ansatz_part(self, names)
        return self

    def as_encoder(self):
        """
        Set all the parameters as encoder.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> pr.as_encoder()
            >>> pr.encoder_parameters
            ['b', 'a']
        """
        ParameterResolver_.as_encoder(self)
        return self

    def as_ansatz(self):
        """
        Set all the parameters as ansatz.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> pr.as_encoder()
            >>> pr.as_ansatz()
            >>> pr.ansatz_parameters
            ['a', 'b']
        """
        ParameterResolver_.as_ansatz(self)
        return self

    def astype(self, dtype) -> "ParameterResolver":
        """
        Convert ParameterResolver to different dtype.

        Args:
            dtype (mindquantum.dtype): new data type of parameter resolver you want.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> a = ParameterResolver('a')
            >>> a
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000
            ],
            const: 0.000000
            )
            >>> import mindquantum as mq
            >>> a.astype(mq.complex128)
            ParameterResolver(dtype: complex128,
            data: [
                    a: (1.000000, 0.000000)
            ],
            const: (0.000000, 0.000000)
            )
        """
        _check_mq_type(dtype)
        return ParameterResolver(ParameterResolver_.astype(self, dtype), internal=True)

    def combination(self, other: typing.Union[typing.Dict[str, numbers.Number], "ParameterResolver"]):
        """
        Apply linear combination between this parameter resolver with input parameter resolver.

        Args:
            other (Union[Dict[str, numbers.Number], ParameterResolver]): The parameter resolver you
                want to do linear combination.

        Returns:
            numbers.Number, the combination result.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr1 = ParameterResolver({'a': 1, 'b': 2})
            >>> pr2 = ParameterResolver({'a': 2, 'b': 3})
            >>> pr1.combination(pr2)
            ParameterResolver(dtype: float64, const: 8.000000)
        """
        _check_input_type('other', (ParameterResolver, dict), other)
        const = self.const
        for k, v in self.items():
            if k in other:
                const += v * other[k]
            else:
                raise ValueError(f"{k} not in input parameter resolver.")
        return self.__class__(const)

    @const.setter
    def const(self, value: numbers.Number):
        """Setter method for const."""
        ParameterResolver_.set_const(self, from_numpy(np.array([value])))

    def conjugate(self) -> "ParameterResolver":
        """
        Get the conjugate of the parameter resolver.

        Returns:
            ParameterResolver, the conjugate version of this parameter resolver.

        Examples:
            >>> import mindquantum as mq
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a' : 1, 'b': 1j}, dtype=mq.complex128)
            >>> pr.conjugate().expression()
            'a + (-1j)*b'
        """
        return ParameterResolver(ParameterResolver_.conjugate(self), internal=True)

    def dumps(self, indent=4) -> str:
        """
        Dump ParameterResolver into JSON(JavaScript Object Notation).

        Note:
            Since float32 type value is not serializable, so ParameterResolver with
            mindquantum.float32 and mindquantum.complex64 type is not serializable.

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            string(JSON), the JSON of ParameterResolver

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2}, const=3 + 4j)
            >>> pr.no_grad_part('a', 'b')
            >>> print(pr.dumps())
            {
                "pr_data": {
                    "a": [
                        1.0,
                        0.0
                    ],
                    "b": [
                        2.0,
                        0.0
                    ]
                },
                "const": [
                    3.0,
                    4.0
                ],
                "dtype": "mindquantum.complex128",
                "no_grad_parameters": [
                    "b",
                    "a"
                ],
                "encoder_parameters": []
            }
        """
        if indent is not None:
            _check_int_type('indent', indent)
        dic = {}
        dic['dtype'] = str(self.dtype)
        if mq.is_single_precision(self.dtype):
            double_version = self.astype(mq.to_double_precision(self.dtype))
        else:
            double_version = self
        dic['pr_data'] = {i: (j.real, j.imag) for i, j in double_version.items()}
        dic['const'] = (double_version.const.real, double_version.const.imag)
        dic['no_grad_parameters'] = list(double_version.no_grad_parameters)
        dic['encoder_parameters'] = list(double_version.encoder_parameters)
        return json.dumps(dic, indent=indent)

    def encoder_part(self, *names):
        """
        Set which part is encoder parameters.

        Args:
            names (tuple[str]): Parameters that will be serve as encoder.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.encoder_part('a')
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000,
                    b: 2.000000
            ],
            const: 0.000000,
            encoder parameters: {a, }
            )
            >>> pr.encoder_parameters
            ['a']
        """
        for name in names:
            _check_input_type('name', str, name)
        ParameterResolver_.encoder_part(self, names)
        return self

    def expression(self) -> str:
        """
        Get the expression string of this parameter resolver.

        Returns:
            str, the string expression of this parameter resolver.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> import numpy as np
            >>> pr = PR({'a': np.pi}, np.sqrt(2))
            >>> pr.expression()
            'π*a + √2'
        """
        string = {}
        for k, v in self.items():
            expr = string_expression(v)
            string[k] = expr
            if expr == '1':
                string[k] = ''
            if expr == '-1':
                string[k] = '-'

        string[''] = string_expression(self.const)
        res = ''
        for k, v in string.items():
            current_s = v
            if current_s.endswith('j'):
                current_s = f'({current_s})'
            if res and (current_s.startswith('(') or not current_s.startswith('-')):
                res += ' + '
            if current_s != '0':
                res += join_without_empty('' if current_s == '-' else '*', [current_s, k])
        if res.endswith(' + '):
            res = res[:-3]
        return res if res else '0'

    def items(self) -> typing.Generator[str, numbers.Number, None]:
        """
        Return an iterator that yields the name and value of all parameters.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 0, 'b': 1})
            >>> list(a.items())
            [('a', 0.0), ('b', 1.0)]
        """
        for k, v in ParameterResolver_.params_data(self).items():
            yield (k, np.array(v)[0])

    def is_anti_hermitian(self) -> bool:
        """
        To check whether the parameter value of this parameter resolver is anti hermitian or not.

        Returns:
            bool, whether the parameter resolver is anti hermitian or not.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1})
            >>> pr.is_anti_hermitian()
            False
            >>> (pr + 3).is_anti_hermitian()
            False
            >>> (pr*1j).is_anti_hermitian()
            True
        """
        return ParameterResolver_.is_anti_hermitian(self)

    def is_const(self) -> bool:
        """
        Check whether this parameter resolver represents a constant number.

        This means that there is no parameter with non zero coefficient in this parameter resolver.

        Returns:
            bool, whether this parameter resolver represent a constant number.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR(1.0)
            >>> pr.is_const()
            True
        """
        return ParameterResolver_.is_const(self)

    def is_hermitian(self) -> bool:
        """
        To check whether the parameter value of this parameter resolver is hermitian or not.

        Returns:
            bool, whether the parameter resolver is hermitian or not.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1})
            >>> pr.is_hermitian()
            True
            >>> (pr + 3).is_hermitian()
            True
            >>> (pr * 1j).is_hermitian()
            False
        """
        return ParameterResolver_.is_hermitian(self)

    def keys(self) -> typing.Generator[str, None, None]:
        """
        Return an iterator that yields the name and value of all parameters.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 0, 'b': 1})
            >>> list(a.keys())
            ['a', 'b']
        """
        yield from ParameterResolver_.params_name(self)

    def no_grad(self):
        """
        Set all parameters to not require gradient calculation. Inplace operation.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad()
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000,
                    b: 2.000000
            ],
            const: 0.000000,
            no grad parameters: {a, b}
            )
            >>> pr.requires_grad_parameters
            []
        """
        ParameterResolver_.no_grad(self)
        return self

    def no_grad_part(self, *names):
        """
        Set part of parameters that not requires grad.

        Args:
            names (tuple[str]): Parameters that not requires grad.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad_part('a')
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000,
                    b: 2.000000
            ],
            const: 0.000000,
            no grad parameters: {a, }
            )
            >>> pr.requires_grad_parameters
            ['b']
        """
        for name in names:
            _check_input_type('name', str, name)
        ParameterResolver_.no_grad_part(self, names)
        return self

    def pop(self, v: str) -> numbers.Number:
        """
        Pop out a parameter.

        Args:
            v (str): The parameter you want to pop.

        Returns:
            numbers.Number, the popped out parameter value.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.pop('a')
            1.0
        """
        return np.array(ParameterResolver_.pop(self, v))[0]

    def requires_grad(self):
        """
        Set all parameters of this parameter resolver to require gradient calculation.

        Inplace operation.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad_part('a')
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000,
                    b: 2.000000
            ],
            const: 0.000000,
            no grad parameters: {a, }
            )
            >>> pr.requires_grad()
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000,
                    b: 2.000000
            ],
            const: 0.000000
            )
            >>> pr.requires_grad_parameters
            ['a', 'b']
        """
        ParameterResolver_.requires_grad(self)
        return self

    def requires_grad_part(self, *names):
        """
        Set part of parameters that requires grad. Inplace operation.

        Args:
            names (tuple[str]): Parameters that requires grad.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad()
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000,
                    b: 2.000000
            ],
            const: 0.000000,
            no grad parameters: {a, b}
            )
            >>> pr.requires_grad_part('a')
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000,
                    b: 2.000000
            ],
            const: 0.000000,
            no grad parameters: {b, }
            )
            >>> pr.requires_grad_parameters
            ['a']
        """
        for name in names:
            _check_input_type('name', str, name)
        ParameterResolver_.requires_grad_part(self, names)
        return self

    def subs(self, other: typing.Union["ParameterResolver", typing.Dict[str, numbers.Number]]):
        """
        Substitute the variable value to ParameterResolver.

        Args:
            other (Union[ParameterResolver, Dict[str, numbers.Number]]): the value of variables in parameter resolver.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1.2, 'b': 2}, 3.4)
            >>> a.expression()
            '1.2*a + 2*b + 3.4'
            >>> a.subs({'a': 0.3})
            >>> a.expression()
            '2*b + 3.76'
        """
        if not isinstance(other, ParameterResolver):
            other = ParameterResolver(other)
        ParameterResolver_.subs(self, other)

    def to_real_obj(self) -> "ParameterResolver":
        """Convert to real type."""
        return self.astype(mq.to_real_type(self.dtype))

    def update(self, other: "ParameterResolver"):
        """
        Update this parameter resolver with other parameter resolver.

        Args:
            other (ParameterResolver): other parameter resolver.

        Raises:
            ValueError: If some parameters require grad and not require grad in other parameter resolver and vice versa
                and some parameters are encoder parameters and not encoder in other parameter resolver and vice versa.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr1 = ParameterResolver({'a': 1})
            >>> pr2 = ParameterResolver({'b': 2})
            >>> pr2.no_grad()
            ParameterResolver(dtype: float64,
            data: [
                    b: 2.000000
            ],
            const: 0.000000,
            no grad parameters: {b}
            )
            >>> pr1.update(pr2)
            >>> pr1
            ParameterResolver(dtype: float64,
            data: [
                    a: 1.000000,
                    b: 2.000000
            ],
            const: 0.000000,
            no grad parameters: {b, }
            )
            >>> pr1.no_grad_parameters
            ['b']
        """
        _check_input_type('other', ParameterResolver, other)
        ParameterResolver_.update(self, other)

    def values(self) -> typing.Generator[numbers.Number, None, None]:
        """
        Return an iterator that yields the name and value of all parameters.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 0, 'b': 1})
            >>> list(a.values())
            [0.0, 1.0]
        """
        for v in ParameterResolver_.params_value(self):
            yield np.array(v)[0]
