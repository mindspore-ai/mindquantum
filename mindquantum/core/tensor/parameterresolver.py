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
        for k in ParameterResolver_.params_name(self):
            yield k

    def values(self):
        for v in ParameterResolver_.params_value(self):
            yield np.array(v)[0]

    def items(self):
        for k, v in ParameterResolver_.params_data(self).items():
            yield (k, np.array(v)[0])

    def __contains__(self, key: str) -> bool:
        return ParameterResolver_.__contains__(key)

    def __copy__(self) -> "ParameterResolver":
        return ParameterResolver(self, internal=True)

    def __deepcopy__(self) -> "ParameterResolver":
        return ParameterResolver(self, internal=True)

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
        return list(self.values())

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
        return [i for i in self.params_name if i in self._cpp_obj.no_grad_parameters]

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
        return [i for i in self.params_name if i in self._cpp_obj.encoder_parameters]

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



if __name__ == "__main__":
    print(ParameterResolver({"a": 3.0}))
    print(ParameterResolver({"a": 3.0}, dtype=mqtype.complex128))
    print(ParameterResolver({"a": 3.0}, 3.4j))
    print(ParameterResolver("a"))
    print(ParameterResolver("a", dtype=mqtype.complex128))
    print(ParameterResolver(1.0))
    print(ParameterResolver(1.0j))
    a = ParameterResolver('a', 1.0, dtype=mqtype.complex128)
    print(a)
    print(a.astype(mqtype.float32))
    print(a.dtype())
