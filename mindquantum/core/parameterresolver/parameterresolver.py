# -*- coding: utf-8 -*-
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
"""Parameter resolver."""

import numbers
import copy
import json
from typing import Iterable
import numpy as np
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.utils.type_value_check import _check_int_type
from mindquantum.utils.type_value_check import _check_np_dtype
from mindquantum.utils.f import is_two_number_close
from mindquantum.utils.f import string_expression
from mindquantum.utils.f import join_without_empty
from mindquantum import mqbackend as mb


def is_type_upgrade(origin_v, other_v):
    """check whether type upgraded."""
    tmp = origin_v + other_v
    return not isinstance(tmp, type(origin_v))


class ParameterResolver:
    """
    A ParameterRsolver can set the parameter of parameterized quantum gate or
    parameterized quantum circuit.

    By specific which part of parameters needs to calculate gradient, the PQC
    operator can only calculate gradient of these parameters.

    Args:
        data (Union[dict, numbers.Number, str, ParameterResolver]): initial parameter names and
            its values. If data is a dict, the key will be the parameter name
            and the value will be the parameter value. If data is a number, this
            number will be the constant value of this parameter resolver. If data
            is a string, then this string will be the only parameter with coefficient
            be 1. Default: None.
        const (number.Number): the constant part of this parameter resolver.
            Default: None.
        dtype (type): the value type of this parameter resolver. Default: numpy.float64.

    Examples:
        >>> from mindquantum.core import ParameterResolver
        >>> pr = ParameterResolver({'a': 0.3})
        >>> pr['b'] = 0.5
        >>> pr.no_grad_part('a')
        {'a': 0.3, 'b': 0.5}, const: 0.0
        >>> pr *= 2
        >>> pr
        {'a': 0.6, 'b': 1.0}, const: 0.0
        >>> pr.expression()
        '3/5*a + b'
        >>> pr.const = 0.5
        >>> pr.expression()
        '3/5*a + b + 1/2'
        >>> pr.no_grad_parameters
        {'a'}
        >>> ParameterResolver(3)
        {}, const: 3.0
        >>> ParameterResolver('a')
        {'a': 1.0}, const: 0.0
    """
    def __init__(self, data=None, const=None, dtype=np.float64):
        _check_np_dtype(dtype)
        self.dtype = dtype
        self.data = {}
        self._const = self.dtype(0)
        if data is None:
            data = {}
        if isinstance(data, numbers.Number):
            if const is not None:
                raise ValueError(f"data and const cannot not both be number.")
            const = data
            data = {}
        if isinstance(data, str):
            data = {data: self.dtype(1)}
        if isinstance(data, self.__class__):
            self.dtype = data.dtype
            self.data = {k: v for k, v in data.items()}
            self.const = data.const
            self.no_grad_parameters = copy.deepcopy(data.no_grad_parameters)
            self.encoder_parameters = copy.deepcopy(data.encoder_parameters)
        elif isinstance(data, dict):
            for k, v in data.items():
                _check_input_type("parameter name", str, k)
                _check_input_type("parameter value", numbers.Number, v)
                if not k.strip():
                    raise KeyError(f"parameter name cannot be empty string.")

            for k, v in data.items():
                self[k] = self.dtype(v)
            if const is None:
                const = 0
            _check_input_type("const", numbers.Number, const)
            self.const = self.dtype(const)
            self.no_grad_parameters = set()
            self.encoder_parameters = set()

    def astype(self, dtype, inplace=False):
        """
        Change the data type of this parameter resolver.

        Args:
            dtype (type): The type of data.
            inplace (bool): Whether to change the type inplace.

        Returns:
            ParameterResolver, the parameter resolver with given data type.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> import numpy as np
            >>> pr = PR({'a': 1.0}, 2.0)
            >>> pr
            {'a': 1.0}, const: 2.0
            >>> pr.astype(np.complex128, inplace=True)
            >>> pr
            {'a': (1+0j)}, const: (2+0j)
        """
        _check_np_dtype(dtype)
        _check_input_type('inplace', bool, inplace)
        if inplace:
            if dtype != self.dtype:
                self.dtype = dtype
                for k in self.keys():
                    self[k] = dtype(self[k])
                self.const = dtype(self.const)
            return self
        new = copy.copy(self)
        new.astype(dtype, inplace=True)
        return new

    @property
    def const(self):
        """
        Get the constant part of this parameter resolver.

        Returns:
            numbers.Number, the constant part of this parameter resolver.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1}, 2.5)
            >>> pr.const
            2.5
        """
        return self._const

    @const.setter
    def const(self, const_value):
        """
        The setter method of const.
        """
        _check_input_type('const value', numbers.Number, const_value)
        if is_type_upgrade(self.const, const_value):
            self.astype(type(const_value), True)
        self._const = self.dtype(const_value)

    def __len__(self):
        """
        Get the number of parameters in this parameter resolver. Please note that
        the parameter with 0 coefficient is also considered.

        Returns:
            int, the number of all parameters.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> a = PR({'a': 0, 'b': 1})
            >>> a.expression()
            'b'
            >>> len(a)
            2
        """
        return len(self.data)

    def keys(self):
        """
        A iterator that yield the name of all parameters.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> a = PR({'a': 0, 'b': 1})
            >>> list(a.keys())
            ['a', 'b']
        """
        for k in self.data.keys():
            yield k

    def values(self):
        """
        A iterator that yield the value of all parameters.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> a = PR({'a': 0, 'b': 1})
            >>> list(a.values())
            [0.0, 1.0]
        """
        for v in self.data.values():
            yield v

    def items(self):
        """
        A iterator that yield the name and value of all parameters.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> a = PR({'a': 0, 'b': 1})
            >>> list(a.items())
            [('a', 0.0), ('b', 1.0)]
        """
        for k, v in self.data.items():
            yield (k, v)

    def is_const(self):
        """
        Check that whether this parameter resolver represent a constant number, which
        means that there is no non zero parameter in this parameter resolver.

        Returns:
            bool, whether this parameter resolver represent a constant number.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR(1.0)
            >>> pr.is_const()
            True
        """
        if not self.data:
            return True
        for v in self.values():
            if not is_two_number_close(v, 0):
                return False
        return True

    def __bool__(self):
        """
        Check whether this parameter resolver has non zero constant or parameter
        with non zero coefficient.

        Returns:
            bool, False if this parameter resolver represent zero and True if not.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR(0)
            >>> bool(pr)
            False
        """
        return not (self.is_const() and is_two_number_close(self.const, 0))

    def __setitem__(self, keys, values):
        """
        Set the value of parameter in this parameter resolver. You can set multiple
        values of multiple parameters with given iterable keys and values.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR(0)
            >>> pr['a'] = 2.5
            >>> pr.expression()
            '5/2*a'
        """
        if isinstance(keys, str):
            _check_input_type("parameter name", str, keys)
            _check_input_type("parameter value", numbers.Number, values)
            if not keys.strip():
                raise KeyError(f"parameter name cannot be empty string.")
            if is_type_upgrade(self.dtype(0), values):
                self.astype(type(values), True)
            self.data[keys] = self.dtype(values)
        elif isinstance(keys, Iterable):
            if not isinstance(values, Iterable):
                raise ValueError("Values should be iterable.")
            if len(values) != len(keys):
                raise ValueError("Size of keys and values do not match.")
            for k, v in zip(keys, values):
                self[k] = v
        else:
            raise TypeError("Parameter name should be a string, but get {}!".format(type(keys)))

    def __getitem__(self, key):
        """
        Get the parameter value from this parameter resolver.

        Returns:
            numbers.Number, the parameter value.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> pr['a']
            1.0
        """
        if key not in self.data:
            raise KeyError(f"parameter {key} not in this parameter resolver")
        return self.data[key]

    def __iter__(self):
        """
        Yield the parameter name.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> list(pr)
            ['a', 'b']
        """
        for i in self.data.keys():
            yield i

    def __contains__(self, key):
        """
        Check whether the given key is in this parameter resolver or not.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> 'c' in pr
            False
        """
        return key in self.data

    def get_cpp_obj(self):
        """Get the cpp object of this parameter resolver."""
        is_const = self.is_const()
        const = self.const
        cpp = mb.parameter_resolver(self.data, self.no_grad_parameters,
                                    set(self.params_name) - self.no_grad_parameters, self.encoder_parameters,
                                    set(self.params_name) - self.encoder_parameters, const, is_const)
        self.const = const
        return cpp

    def __eq__(self, other):
        """
        To check whether two parameter resolvers are equal.

        Args:
            other (Union[numbers.Number, str, ParameterResolver]): The parameter resolver
                or number you want to compare. If a number or string is given, this number will
                convert to a parameter resolver.

        Returns:
            bool, whether two parameter resolvers are equal.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> PR(3) == 3
            True
            >>> PR('a') == 3
            False
            >>> PR({'a': 2}, 3) == PR({'a': 2}) + 3
            True
        """
        if isinstance(other, numbers.Number):
            if not self.is_const():
                return False
            return is_two_number_close(self.const, other)
        if isinstance(other, str):
            if not is_two_number_close(self.const, 0):
                return False
            if len(self.data) == 1 and other in self.data:
                return True
            return False
        _check_input_type("other", ParameterResolver, other)
        if not is_two_number_close(self.const, other.const):
            return False
        if self.no_grad_parameters != other.no_grad_parameters:
            return False
        if set(self.data.keys()) != set(other.data.keys()):
            return False
        if self.encoder_parameters != other.encoder_parameters:
            return False
        for k, v in self.items():
            if not is_two_number_close(v, other[k]):
                return False
        return True

    def __copy__(self):
        """
        Copy a new parameter resolver.

        Returns:
            ParameterResolver, the new parameter resolver you copied.

        Examples:
            >>> import copy
            >>> from mindquantum.core import ParameterResolver as PR
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
        pr = ParameterResolver()
        pr.dtype = self.dtype
        pr.data = {k: v for k, v in self.items()}
        pr.const = self.const
        pr.no_grad_parameters = {i for i in self.no_grad_parameters}
        pr.encoder_parameters = {i for i in self.encoder_parameters}
        return pr

    def __iadd__(self, other):
        """
        Inplace add a number or parameter resolver.

        Args (Union[numbers.Number, ParameterResolver]): the number or parameter
            resolver you want add.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 3})
            >>> pr += 4
            >>> pr.expression()
            '3*a + 4'
            >>> pr += PR({'b': 1.5})
            >>> pr
            '3*a + 3/2*b + 4'
        """
        _check_input_type('other', (numbers.Number, ParameterResolver), other)
        if isinstance(other, numbers.Number):
            self.const += other
        if isinstance(other, ParameterResolver):
            self.const += other.const
            for k, v in other.data.items():
                if k in self.data:
                    if k in self.no_grad_parameters and k not in other.no_grad_parameters \
                        or k in other.no_grad_parameters and k not in self.no_grad_parameters:
                        raise RuntimeError(f"gradient property of parameter {k} conflict.")
                    if k in self.encoder_parameters and k not in other.encoder_parameters \
                        or k in other.encoder_parameters and k not in self.encoder_parameters:
                        raise RuntimeError(f"encoder or ansatz property of parameter {k} conflict.")
                    self[k] += v
                else:
                    self[k] = v
                    if k in other.no_grad_parameters:
                        self.no_grad_parameters.add(k)
                    if k in other.encoder_parameters:
                        self.encoder_parameters.add(k)
        return self

    def __add__(self, other):
        """
        Add a number or parameter resolver.

        Args:
            other (Union[numbers.Number, ParameterResolver])

        Returns:
            ParameterResolver, the result of add.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 3})
            >>> pr + 1
            {'a': 3.0}, const: 1.0
            >>> pr + pr
            {'a': 6.0}, const: 0.0
        """
        res = copy.copy(self)
        res += other
        return res

    def __radd__(self, other):
        """
        Add a number or ParameterResolver.
        """
        return self + other

    def __isub__(self, other):
        """Self subtract a number or ParameterResolver."""
        self += (-other)
        return self

    def __sub__(self, other):
        """Subtract a number or ParameterResolver."""
        _check_input_type('other', (numbers.Number, ParameterResolver), other)
        if isinstance(other, numbers.Number):
            return self + (-other)
        if isinstance(other, ParameterResolver):
            other = copy.copy(other)
            for k in other.data:
                other.data[k] *= -1
            other.const *= -1
            return self + other
        raise TypeError(f"unsupported operand type(s) for -: 'ParameterResolver' and '{type(other)}'")

    def __rsub__(self, other):
        """Self subtract a number or ParameterResolver."""
        return other + (-self)

    def __imul__(self, other):
        """Self multiply a number or ParameterResolver."""
        if isinstance(other, numbers.Number):
            for k in list(self):
                self[k] *= other
            self.const *= other
            return self
        if isinstance(other, self.__class__):
            if self.is_const():
                for k, v in other.items():
                    self[k] = v * self.const
                self.const *= other.const
                return self
            if other.is_const():
                for k in list(self):
                    self[k] *= other.const
                self.const *= other.const
                return self
            raise ValueError("Parameter resolver only support first order variable.")
        raise ValueError(f"other requires a number or a number or a parameter resolver, but get {type(other)}")

    def __mul__(self, other):
        """Multiply a number or ParameterResolver."""
        res = copy.copy(self)
        res *= other
        return res

    def __rmul__(self, other):
        """Multiply a number or ParameterResolver."""
        return self * other

    def __neg__(self):
        """The negative of this parameter resolver."""
        out = -1 * self
        return out

    def __itruediv__(self, other):
        """Divide a number inplace."""
        _check_input_type("other", numbers.Number, other)
        self *= (1 / other)
        return self

    def __truediv__(self, other):
        """Divide a number."""
        res = copy.copy(self)
        res /= other
        return res

    def __str__(self):
        """String expression of this parameter resolver."""
        return f'{str(self.data)}, const: {self.const}'

    def __repr__(self) -> str:
        """Repr of this parameter resolver."""
        return self.__str__()

    def __float__(self):
        """Convert the constant part to float. Raise error if it's not constant."""
        if not self.is_const():
            raise ValueError("parameter resolver is not constant, cannot convert to float.")
        return float(self.const)

    def expression(self):
        """
        Get the expression string of this parameter resolver.

        Returns:
            str, the string expression of this parameter resolver.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> import numpy as np
            >>> pr = PR({'a': np.pi}, np.sqrt(2))
            >>> pr.expression()
            'π*a + √2'
        """
        s = {}
        for k, v in self.data.items():
            s[k] = string_expression(v)
            if s[k] == '1':
                s[k] = ''
            if s[k] == '-1':
                s[k] = '-'

        const = string_expression(self.const)
        s[''] = const
        res = ''
        for k, v in s.items():
            current_s = s[k]
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
    def para_value(self):
        """
        Get the parameters value.

        Returns:
            list, a list of parameters value.

        Examples:
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.para_value
            [1, 2]
        """
        return np.array(list(self.values()), dtype=self.dtype)

    def requires_grad(self):
        """
        Set all parameters of this parameter resolver to require gradient
        calculation. Inplace operation.

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
        self.no_grad_parameters = set()
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
        self.no_grad_parameters = set(self.data.keys())
        self.no_grad_parameters.discard('')
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
            if name not in self.data or name == '':
                raise KeyError(f"Parameter {name} not in this parameter resolver!")
            self.no_grad_parameters.discard(name)
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
            if name not in self.data or name == '':
                raise KeyError(f"Parameter {name} not in this parameter resolver!")
            self.no_grad_parameters.add(name)
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
            if name not in self.data or name == '':
                raise KeyError(f"Parameter {name} not in this parameter resolver!")
            self.encoder_parameters.add(name)
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
            if name not in self.data or name == '':
                raise KeyError(f"Parameter {name} not in this parameter resolver!")
            self.encoder_parameters.discard(name)

    def as_encoder(self):
        """
        Set all the parameters as encoder.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> pr.as_encoder()
            >>> pr.encoder_parameters
            {'a', 'b'}
        """
        for name in self.data:
            if name != '':
                self.encoder_parameters.add(name)
        return self

    def as_ansatz(self):
        """
        Set all the parameters as ansatz.

        Returns:
            ParameterResolver, the parameter resolver itself.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> pr.as_encoder()
            >>> pr.as_ansatz()
            >>> pr.ansatz_parameters
            {'a', 'b'}
        """
        for name in self.data:
            self.encoder_parameters.discard(name)
        return self

    def update(self, other):
        """
        Update this parameter resolver with other parameter resolver.

        Args:
            others (ParameterResolver): other parameter resolver.

        Raises:
            ValueError: If some parameters require grad and not require grad in
                other parameter resolver and vice versa and some parameters are encoder
                parameters and not encoder in other parameter resolver and vice versa.

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
        for k, v in other.items():
            if k in other.no_grad_parameters and (k in self.data and k not in self.no_grad_parameters) or \
                (k not in other.no_grad_parameters and k in self.no_grad_parameters):
                raise ValueError(f"Parameter conflict, {k} require grad in some parameter resolver and not \
require grad in other parameter resolver.")
            if k in other.encoder_parameters and (k in self.data and k not in self.encoder_parameters) or \
                (k not in other.encoder_parameters and k in self.encoder_parameters):
                raise ValueError(f"Parameter conflict, {k} is encoder parameter in some parameter resolver and is not \
encoder parameter in other parameter resolver.")
            self[k] = v
            if k in other.no_grad_parameters:
                self.no_grad_parameters.add(k)
            if k in other.encoder_parameters:
                self.encoder_parameters.add(k)

    @property
    def requires_grad_parameters(self):
        """
        Get parameters that requires grad.

        Returns:
            set, the set of parameters that requires grad.

        >>> from mindquantum.core import ParameterResolver as PR
        >>> a = PR({'a': 1, 'b': 2})
        >>> a.requires_grad_parameters
        {'a', 'b'}
        """
        return set(self.params_name) - self.no_grad_parameters

    @property
    def ansatz_parameters(self):
        """
        Get parameters that is ansatz parameters.

        Returns:
            set, the set of ansatz parameters.

        >>> from mindquantum.core import ParameterResolver as PR
        >>> a = PR({'a': 1, 'b': 2})
        >>> a.ansatz_parameters
        {'a', 'b'}
        """
        return set(self.params_name) - self.encoder_parameters

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
        res = copy.copy(self)
        for k, v in res.data.items():
            res.data[k] = np.conj(v)
        res.const = np.conj(self.const)
        return res

    def combination(self, other):
        """
        Apply linear combination between this parameter resolver with input pr.

        Args:
            pr (Union[dict, ParameterResolver]): The parameter resolver you
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
        c = self.const
        for k, v in self.items():
            if k in other:
                c += v * other[k]
            else:
                raise ValueError(f"{k} not in input parameter resolver.")
        return self.__class__(c, dtype=type(c))

    def pop(self, v):
        """
        Pop out a parameter.

        Returns:
            numbers.Number, the popped out parameter value.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> a = PR({'a': 1, 'b': 2})
            >>> a.pop('a')
            1.0
        """
        out = self.data.pop(v)
        self.encoder_parameters.discard(v)
        self.no_grad_parameters.discard(v)
        return out

    @property
    def real(self):
        """
        Get the real part of every parameter value.

        Returns:
            ParameterResolver, real part parameter value.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR('a', 3) + 1j * PR('a', 4)
            >>> pr
            {'a': (1+1j)}, const: (3+4j)
            >>> pr.real
            {'a': 1.0}, const: 3.0
        """
        out = self.__class__()
        for k, v in self.data.items():
            r_v = np.real(v)
            out[k] = r_v
            if k in self.no_grad_parameters:
                out.no_grad_parameters.add(k)
            if k in self.encoder_parameters:
                out.encoder_parameters.add(k)
        out.const = np.real(self.const)
        return out

    @property
    def imag(self):
        """
        Get the image part of every parameter value.

        Returns:
            ParameterResolver, image part parameter value.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR('a', 3) + 1j * PR('a', 4)
            >>> pr
            {'a': (1+1j)}, const: (3+4j)
            >>> pr.imag
            {'a': 1.0}, const: 4.0
        """
        out = self.__class__()
        for k, v in self.data.items():
            i_v = np.imag(v)
            out[k] = i_v
            if k in self.no_grad_parameters:
                out.no_grad_parameters.add(k)
            if k in self.encoder_parameters:
                out.encoder_parameters.add(k)
        out.const = np.imag(self.const)
        return out

    def is_hermitian(self):
        """
        To check whether the parameter value of this parameter resolver is
        hermitian or not.

        Returns:
            bool, whether the parameter resolver is hermitian or not.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1})
            >>> pr.is_hermitian()
            True
            >>> (pr + 3).is_hermitian()
            True
            >>> (pr * 1j).is_hermitian()
            False
        """
        return self == self.conjugate()

    def is_anti_hermitian(self):
        """
        To check whether the parameter value of this parameter resolver is
        anti hermitian or not.

        Returns:
            bool, whether the parameter resolver is anti hermitian or not.

        Examples:
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1})
            >>> pr.is_anti_hermitian()
            False
            >>> (pr + 3).is_anti_hermitian()
            False
            >>> (pr*1j).is_anti_hermitian()
            True
        """
        return self == -self.conjugate()

    def dumps(self, indent=4):
        '''
        Dump ParameterResolver into JSON(JavaScript Object Notation)

        Args:
            indent (int): Then JSON array elements and object members will be
                pretty-printed with that indent level. Default: 4.

        Returns:
            string(JSON), the JSON of ParameterResolver

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2, 'c': 3, 'd': 4})
            >>> pr.no_grad_part('a', 'b')
            >>> print(pr.dumps())
            {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "__class__": "ParameterResolver",
                "__module__": "parameterresolver",
                "no_grad_parameters": [
                    "a",
                    "b"
                ]
            }
        '''
        if indent is not None:
            _check_int_type('indent', indent)
        dic = dict(zip(self.params_name, self.para_value))
        dic['__class__'] = self.__class__.__name__
        dic['__module__'] = self.__module__

        dic['no_grad_parameters'] = list()
        for j in self.no_grad_parameters:
            dic["no_grad_parameters"].append(j)
        dic["no_grad_parameters"].sort()

        return json.dumps(dic, indent=indent)

    @staticmethod
    def loads(strs):
        '''
        Load JSON(JavaScript Object Notation) into FermionOperator

        Args:
            strs (str): The dumped parameter resolver string.

        Returns:
            FermionOperator, the FermionOperator load from strings

        Examples:
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> strings = """
            ...     {
            ...         "a": 1,
            ...         "b": 2,
            ...         "c": 3,
            ...         "d": 4,
            ...         "__class__": "ParameterResolver",
            ...         "__module__": "parameterresolver",
            ...         "no_grad_parameters": [
            ...             "a",
            ...             "b"
            ...         ]
            ...     }
            ...     """
            >>> obj = ParameterResolver.loads(string)
            >>> print(obj)
            {'a': 1, 'b': 2, 'c': 3, 'd': 4}
            >>> print('requires_grad_parameters is:', obj.requires_grad_parameters)
            requires_grad_parameters is: {'c', 'd'}
            >>> print('no_grad_parameters is :', obj.no_grad_parameters)
            no_grad_parameters is : {'b', 'a'}
        '''
        _check_input_type('strs', str, strs)
        dic = json.loads(strs)

        if '__class__' in dic:
            class_name = dic.pop('__class__')

            if class_name == 'ParameterResolver':
                module_name = dic.pop('__module__')
                module = __import__(module_name)
                class_ = getattr(module, class_name)
                no_grad_parameters_list = dic.pop('no_grad_parameters')

                args = dic
                p = class_(args)

                for i in no_grad_parameters_list:
                    p.no_grad_part(str(i))

            else:
                raise TypeError("Require a ParameterResolver class, but get {} class".format(class_name))

        else:
            raise ValueError("Expect a '__class__' in strings, but not found")

        return p
