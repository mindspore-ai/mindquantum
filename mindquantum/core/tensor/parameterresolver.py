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

import numpy as np

from mindquantum._math.pr import ParameterResolver as ParameterResolver_
from mindquantum._math.tensor import Tensor as Tensor_
from mindquantum.core.tensor import dtype as mqtype
from mindquantum.utils.string_utils import join_without_empty, string_expression
from mindquantum.utils.type_value_check import _check_input_type, _check_int_type


class ParameterResolver(ParameterResolver_):
    """
    ParameterResolver('a'[, dtype=mq.float32])
    ParameterResolver(1.0[, dtype=mq.float32])
    ParameterResolver({'a': 2}[, const=1.2, dtype=mq.float32])
    """

    def __init__(self, data=None, const=None, dtype=None, internal=False):
        if isinstance(data, ParameterResolver):
            internal = True
        if internal:
            if dtype != None:
                ParameterResolver_.__init__(self, data.astype(dtype))
            else:
                ParameterResolver_.__init__(self, data)
        else:
            if isinstance(data, str):
                if dtype is None:
                    dtype = mqtype.float64
                    if const is not None:
                        if isinstance(const, numbers.Number) and not isinstance(const, numbers.Real):
                            dtype = mqtype.complex128
                if const is None:
                    const = Tensor_(0.0, dtype)
                else:
                    const = Tensor_(const, dtype)
                ParameterResolver_.__init__(self, data, const, dtype)  # PR('a'[, 1.0, mq.float64])
            elif isinstance(data, dict):
                if dtype is None:
                    dtype = mqtype.float64
                    for v in data.values():
                        if isinstance(v, numbers.Number) and not isinstance(v, numbers.Real):
                            dtype = mqtype.complex128
                            break
                    if const is not None:
                        if isinstance(const, numbers.Number) and not isinstance(const, numbers.Real):
                            dtype = mqtype.complex128
                if const is None:
                    const = Tensor_(0.0, dtype)
                else:
                    const = Tensor_(const, dtype)
                # PR({'a': 1.0}[, 2.0, mq.float64])
                ParameterResolver_.__init__(self, {i: Tensor_(j, dtype) for i, j in data.items()}, const, dtype)
            elif isinstance(data, numbers.Number):
                if dtype is None:
                    dtype = mqtype.float64
                    if isinstance(data, numbers.Number) and not isinstance(data, numbers.Real):
                        dtype = mqtype.complex128
                ParameterResolver_.__init__(self, Tensor_(data, dtype), dtype)  # PR(1.0[, mq.float64])

    def __str__(self) -> str:
        return self.expression()

    def __repr__(self) -> str:
        return ParameterResolver_.__str__(self)

    def astype(self, dtype) -> "ParameterResolver":
        return ParameterResolver(ParameterResolver_.astype(self, dtype), internal=True)

    @property
    def dtype(self):
        return ParameterResolver_.dtype(self)

    @property
    def const(self) -> numbers.Number:
        return np.array(ParameterResolver_.get_const(self))[0]

    @const.setter
    def const(self, value):
        ParameterResolver_.set_const(self, Tensor_(value))

    def __len__(self) -> int:
        return ParameterResolver_.__len__(self)

    def keys(self):
        yield from ParameterResolver_.params_name(self)

    def values(self):
        for v in ParameterResolver_.params_value(self):
            yield np.array(v)[0]

    def items(self):
        for k, v in ParameterResolver_.params_data(self).items():
            yield (k, np.array(v)[0])

    def __contains__(self, key: str) -> bool:
        return ParameterResolver_.__contains__(self, key)

    def __copy__(self) -> "ParameterResolver":
        return ParameterResolver(self, internal=True)

    def __deepcopy__(self) -> "ParameterResolver":
        return ParameterResolver(self, internal=True)

    def __eq__(self, other) -> bool:
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

    def dumps(self, indent=4):
        """
        Dump ParameterResolver into JSON(JavaScript Object Notation).

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            string(JSON), the JSON of ParameterResolver

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2}, const=3 + 4j, dtype=complex)
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
                "dtype": "complex",
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
        dic['pr_data'] = {i: (j.real, j.imag) for i, j in self.items()}
        dic['const'] = (self.const.real, self.const.imag)
        dic['dtype'] = str(self.dtype)
        dic['no_grad_parameters'] = list(self.no_grad_parameters)
        dic['encoder_parameters'] = list(self.encoder_parameters)
        return json.dumps(dic, indent=indent)

    @staticmethod
    def loads(strs):
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
            {'a': 1, 'b': 2, 'c': 3, 'd': 4}, const: 0
            >>> print('requires_grad_parameters is:', obj.requires_grad_parameters)
            requires_grad_parameters is: {'c', 'd'}
            >>> print('no_grad_parameters is :', obj.no_grad_parameters)
            no_grad_parameters is : {'b', 'a'}
        """
        _check_input_type('strs', str, strs)
        dic = json.loads(strs)
        if 'dtype' not in dic:
            raise ValueError("Invalid string. Cannot convert it to ParameterResolver, no key dtype")
        dtype = mqtype.str_dtype_map[dic['dtype']]
        if dtype in (mqtype.complex128, mqtype.complex64):
            const = dic['const'][0] + 1j * dic['const'][1]
            data = {i: j[0] + j[1] * 1j for i, j in dic['pr_data'].items()}
        else:
            const = dic['const'][0]
            data = {i: j[0] for i, j in dic['pr_data'].items()}
        out = ParameterResolver(data, const, dtype)
        out.no_grad_part(*dic['no_grad_parameters'])
        out.encoder_part(*dic['encoder_parameters'])
        return out

    def is_const(self) -> bool:
        return ParameterResolver_.is_const(self)

    def __bool__(self) -> bool:
        return ParameterResolver_.is_not_zero(self)

    def __setitem__(self, key, value):
        ParameterResolver_.set_item(self, key, Tensor_(value))

    def __getitem__(self, key):
        return np.array(ParameterResolver_.get_item(self, key))[0]

    def __iter__(self):
        """
        Yield the parameter name.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> list(pr)
            ['a', 'b']
        """
        yield from self.keys()

    def __float__(self):
        """Convert the constant part to float. Raise error if it's not constant."""
        if not self.is_const():
            raise ValueError("parameter resolver is not constant, cannot convert to float.")
        return np.float64(self.const)

    def expression(self):
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

    @property
    def params_name(self):
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
    def params_value(self):
        """
        Get the parameters value.

        Returns:
            list, a list of parameters value.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.params_value
            [1, 2]
        """
        return list(self.values())

    def update(self, other):
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
            {'b': 2.0}, const: 0.0
            >>> pr1.update(pr2)
            >>> pr1
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr1.no_grad_parameters
            {'b'}
        """
        _check_input_type('other', ParameterResolver, other)
        ParameterResolver_.update(self, other)

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
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad()
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad_parameters
            {'a', 'b'}
        """
        ParameterResolver_.requires_grad(self)
        return self

    def no_grad(self):
        """
        Set all parameters to not require gradient calculation. Inplace operation.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad()
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad_parameters
            set()
        """
        ParameterResolver_.no_grad(self)
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
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad_part('a')
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad_parameters
            {'a'}
        """
        for name in names:
            _check_input_type('name', str, name)
        ParameterResolver_.requires_grad_part(self, names)
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
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad_parameters
            {'b'}
        """
        for name in names:
            _check_input_type('name', str, name)
        ParameterResolver_.no_grad_part(self, names)
        return self

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
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.encoder_parameters
            {'a'}
        """
        for name in names:
            _check_input_type('name', str, name)
        ParameterResolver_.encoder_part(self, names)
        return self

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
            {'a'}
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
            {'a', 'b'}
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
            {'a', 'b'}
        """
        ParameterResolver_.as_ansatz(self)
        return self

    @property
    def requires_grad_parameters(self):
        """
        Get parameters that requires grad.

        Returns:
            set, the set of parameters that requires grad.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.requires_grad_parameters
            {'a', 'b'}
        """
        return [i for i in self.params_name if i not in self.no_grad_parameters]

    @property
    def no_grad_parameters(self):
        """
        Get parameters that do not require grad.

        Returns:
            set, the set of parameters that do not require grad.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.no_grad()
            >>> a.no_grad_parameters
            {'a', 'b'}
        """
        return list(ParameterResolver_.get_grad_parameters(self))

    @property
    def encoder_parameters(self):
        """
        Get parameters that is encoder parameters.

        Returns:
            set, the set of encoder parameters.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.as_encoder()
            >>> a.encoder_parameters
            {'a', 'b'}
        """
        return list(ParameterResolver_.get_encoder_parameters(self))

    @property
    def ansatz_parameters(self):
        """
        Get parameters that is ansatz parameters.

        Returns:
            set, the set of ansatz parameters.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.ansatz_parameters
            {'a', 'b'}
        """
        return [i for i in self.params_name if i not in self.encoder_parameters]

    def conjugate(self):
        """
        Get the conjugate of the parameter resolver.

        Returns:
            ParameterResolver, the conjugate version of this parameter resolver.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> import numpy as np
            >>> pr = PR({'a' : 1, 'b': 1j}, dtype=np.complex128)
            >>> pr.conjugate().expression()
            'a + (-1j)*b'
        """
        return ParameterResolver(ParameterResolver_.conjugate(self), internal=True)

    def combination(self, other):
        """
        Apply linear combination between this parameter resolver with input parameter resolver.

        Args:
            other (Union[dict, ParameterResolver]): The parameter resolver you
                want to do linear combination.

        Returns:
            numbers.Number, the combination result.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr1 = ParameterResolver({'a': 1, 'b': 2})
            >>> pr2 = ParameterResolver({'a': 2, 'b': 3})
            >>> pr1.combination(pr2)
            {}, const: 8.0
        """
        _check_input_type('other', (ParameterResolver, dict), other)
        const = self.const
        for k, v in self.items():
            if k in other:
                const += v * other[k]
            else:
                raise ValueError(f"{k} not in input parameter resolver.")
        return self.__class__(const)

    def pop(self, v):
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

    @property
    def is_complex(self):
        """Return whether the ParameterResolver instance is currently using complex coefficients."""
        return self.dtype in (mqtype.complex128, mqtype.complex64)

    @property
    def real(self):
        """
        Get the real part of every parameter value.

        Returns:
            ParameterResolver, real part parameter value.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR('a', 3) + 1j * PR('a', 4)
            >>> pr
            {'a': (1+1j)}, const: (3+4j)
            >>> pr.real
            {'a': 1.0}, const: 3.0
        """
        return ParameterResolver(ParameterResolver_.real(self), internal=True)

    @property
    def imag(self):
        """
        Get the imaginary part of every parameter value.

        Returns:
            ParameterResolver, image part parameter value.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR('a', 3) + 1j * PR('a', 4)
            >>> pr
            {'a': (1+1j)}, const: (3+4j)
            >>> pr.imag
            {'a': 1.0}, const: 4.0
        """
        return ParameterResolver(ParameterResolver_.imag(self), internal=True)

    def __iadd__(self, other) -> "ParameterResolver":
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
            '3*a + 3/2*b + 4'
        """
        if isinstance(other, ParameterResolver_):
            ParameterResolver_.__iadd__(self, other)
        else:
            ParameterResolver_.__iadd__(self, ParameterResolver(other))
        return self

    def __add__(self, other) -> "ParameterResolver":
        """
        Add a number or parameter resolver.

        Args:
            other (Union[numbers.Number, ParameterResolver])

        Returns:
            ParameterResolver, the result of add.

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> pr = PR({'a': 3})
            >>> pr + 1
            {'a': 3.0}, const: 1.0
            >>> pr + pr
            {'a': 6.0}, const: 0.0
        """
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(ParameterResolver_.__add__(self, other), internal=True)
        return ParameterResolver(ParameterResolver_.__add__(self, ParameterResolver(other)), internal=True)

    def __radd__(self, other) -> "ParameterResolver":
        """Add a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(other + self, internal=True)
        else:
            return ParameterResolver(ParameterResolver(other) + self, internal=True)

    def __isub__(self, other):
        """Self subtract a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            ParameterResolver_.__isub__(self, other)
        else:
            ParameterResolver_.__isub__(self, ParameterResolver(other))
        return self

    def __sub__(self, other):
        """Subtract a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(ParameterResolver_.__sub__(self, other), internal=True)
        return ParameterResolver(ParameterResolver_.__sub__(self, ParameterResolver(other)), internal=True)

    def __rsub__(self, other):
        """Self subtract a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return other - self
        return ParameterResolver(other) - self

    def __imul__(self, other):
        """Self multiply a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            ParameterResolver_.__imul__(self, other)
        else:
            ParameterResolver_.__imul__(self, ParameterResolver(other))
        return self

    def __mul__(self, other):
        """Multiply a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(ParameterResolver_.__mul__(self, other), internal=True)
        return ParameterResolver(ParameterResolver_.__mul__(self, ParameterResolver(other)), internal=True)

    def __rmul__(self, other):
        """Multiply a number or ParameterResolver."""
        if isinstance(other, ParameterResolver_):
            return other * self
        return ParameterResolver(other) * self

    def __neg__(self):
        """Return the negative of this parameter resolver."""
        return 0 - self

    def __itruediv__(self, other):
        """Divide a number inplace."""
        if isinstance(other, ParameterResolver_):
            ParameterResolver_.__itruediv__(self, other)
        else:
            ParameterResolver_.__itruediv__(self, ParameterResolver(other))
        return self

    def __truediv__(self, other):
        """Divide a number."""
        if isinstance(other, ParameterResolver_):
            return ParameterResolver(ParameterResolver_.__truediv__(self, other), internal=True)
        return ParameterResolver(ParameterResolver_.__truediv__(self, ParameterResolver(other)), internal=True)

    def is_anti_hermitian(self):
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

    def is_hermitian(self):
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
        return ParameterResolver_.anti_hermitian(self)
