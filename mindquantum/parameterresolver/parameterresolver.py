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

from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import sympy as sp


class ParameterResolver(dict):
    """
    A ParameterRsolver can set the parameter of parameterized quantum gate or
    parameterized quantum circuit.

    By specific which part of parameters needs to calculate gradient, the PQC
    operator can only calculate gradient of these parameters.

    Args:
        data (dict): initial parameter names and its values.

    Examples:
        >>> pr = ParameterResolver({'a': 0.3})
        >>> pr['b'] = 0.5
        >>> pr.no_grad_part('a')
        >>> pr *= 2
        >>> pr
        {'a': 0.6, 'b': 1.0}
        >>> pr.no_grad_parameters
        {'a'}
    """
    def __init__(self, data=None):
        if data is None:
            data = {}
        if not isinstance(data, (dict, ParameterResolver)):
            raise TypeError(
                "Require a dict or a ParameterResolver, but get {}!".format(
                    type(data)))
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(
                    "Parameter name should be a string, but get {}!".format(
                        type(k)))
            if not isinstance(v, _num_type):
                raise TypeError(
                    "Require a real number, but get {}, which is {}!".format(
                        v, type(v)))
        super(ParameterResolver, self).__init__(data)
        self.no_grad_parameters = set()
        self.requires_grad_parameters = set(self.para_name)

    def __setitem__(self, keys, values):
        """
        Set parameter or as list of parameters of this parameter resolver.

        By default, the parameter you set requires gradient.

        Args:
            keys (Union[str, list[str]]): The name of parameters.
            values (Union[number, list[number]]): The value of parameters.

        Raises:
            TypeError: If the key that you set is not a string or a iterable of
                string.
        """
        if isinstance(keys, str):
            if not isinstance(values, _num_type):
                raise TypeError(
                    "Parameter value should be a real number, but get {}, which is {}!"
                    .format(values, type(values)))
            super().__setitem__(keys, values)
            self.requires_grad_parameters.add(keys)
        elif isinstance(keys, Iterable):
            if not isinstance(values, Iterable):
                raise ValueError("Values should be iterable.")
            if len(values) != len(keys):
                raise ValueError("Size of keys and values do not match.")
            for i, k in enumerate(keys):
                self.__setitem__(k, values[i])
        else:
            raise TypeError(
                "Parameter name should be a string, but get {}!".format(
                    type(keys)))

    def __imul__(self, num):
        """
        Parameter support inplace multiply.

        Returns:
            :class:`mindquantum.parameterresolver.ParameterResolver`

        Args:
            num (number): Multiply factor.

        Examples:
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr *= 2
            >>> pr
            {'a': 2, 'b': 4}
        """
        no_grad_parameters = deepcopy(self.no_grad_parameters)
        requires_grad_parameters = deepcopy(self.requires_grad_parameters)
        for k in self.keys():
            self[k] = self[k] * num
        self.no_grad_parameters = no_grad_parameters
        self.requires_grad_parameters = requires_grad_parameters
        return self

    def __mul__(self, num):
        """
        Multiply num with every value of parameter resolver.

        Returns:
            :class:`mindquantum.parameterresolver.ParameterResolver`

        Args:
            num (number): Multiply factor.

        Examples:
            >>> pr1 = ParameterResolver({'a': 1, 'b': 2})
            >>> pr2 = pr1 * 2
            >>> pr2
            {'a': 2, 'b': 4}
        """
        no_grad_parameters = deepcopy(self.no_grad_parameters)
        requires_grad_parameters = deepcopy(self.requires_grad_parameters)
        out = deepcopy(self)
        out *= num
        out.no_grad_parameters = no_grad_parameters
        out.requires_grad_parameters = requires_grad_parameters
        return out

    def __rmul__(self, num):
        """
        See :class:`mindquantum.parameterresolver.ParameterResolver.__mul__`.
        """
        return self.__mul__(num)

    def __eq__(self, other):
        _check_pr_type(other)
        no_grad_eq = self.no_grad_parameters == other.no_grad_parameters
        requires_grad_eq = self.requires_grad_parameters == other.requires_grad_parameters
        return super().__eq__(other) and no_grad_eq and requires_grad_eq

    @property
    def para_name(self):
        """
        Get the parameters name.

        Returns:
            list[str], Parameters name.

        Examples:
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.para_name
            ['a', 'b']
        """
        return list(self.keys())

    @property
    def para_value(self):
        """
        Get the parameters value.

        Returns:
            list[float], Parameters value.

        Examples:
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.para_value
            [1, 2]
        """
        return list(self.values())

    def requires_grad(self):
        """
        Set all parameters of this parameter resolver to require gradient
        calculation.

        Returns:
            :class:`mindquantum.parameterresolver.ParameterResolver`

        Examples:
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad_part('a')
            >>> pr.requires_grad()
            >>> pr.requires_grad_parameters
            {'a', 'b'}
        """
        self.no_grad_parameters = set()
        self.requires_grad_parameters = set(self.para_name)
        return self

    def no_grad(self):
        """
        Set all parameters to not require gradient calculation.

        Returns:
            :class:`mindquantum.parameterresolver.ParameterResolver`

        Examples:
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad()
            >>> pr.requires_grad_parameters
            set()
        """
        self.no_grad_parameters = set(self.para_name)
        self.requires_grad_parameters = set()
        return self

    def requires_grad_part(self, *names):
        """
        Set part of parameters that requires grad.

        Args:
            names (tuple[str]): Parameters that requires grad.

        Returns:
            :class:`mindquantum.parameterresolver.ParameterResolver`

        Examples:
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad()
            >>> pr.requires_grad_part('a')
            >>> pr.requires_grad_parameters
            {'a'}
        """
        for name in names:
            if not isinstance(name, str):
                raise TypeError("name should be a string, but get {}!".format(
                    type(name)))
            if name not in self:
                raise KeyError(
                    "Parameter {} not in this parameter resolver!".format(
                        name))
            while name in self.no_grad_parameters:
                self.no_grad_parameters.remove(name)
            while name not in self.requires_grad_parameters:
                self.requires_grad_parameters.add(name)
        return self

    def no_grad_part(self, *names):
        """
        Set part of parameters that not requires grad.

        Args:
            names (tuple[str]): Parameters that not requires grad.

        Returns:
            :class:`mindquantum.parameterresolver.ParameterResolver`

        Examples:
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad_part('a')
            >>> pr.requires_grad_parameters
            {'b'}
        """
        for name in names:
            if not isinstance(name, str):
                raise TypeError("name should be a string, but get {}!".format(
                    type(name)))
            if name not in self:
                raise KeyError(
                    "Parameter {} not in this parameter resolver!".format(
                        name))
            while name not in self.no_grad_parameters:
                self.no_grad_parameters.add(name)
            while name in self.requires_grad_parameters:
                self.requires_grad_parameters.remove(name)
        return self

    def update(self, others):
        """
        Update this parameter resolver with other parameter resolver.

        Args:
            others (:class:`mindquantum.parameterresolver.ParameterResolver`):
                other parameter resolver.

        Raises:
            ValueError: If some parameters require grad and not require grad in
                other parameter resolver and vise versa.

        Examples:
            >>> pr1 = ParameterResolver({'a': 1})
            >>> pr2 = ParameterResolver({'b': 2})
            >>> pr2.no_grad()
            >>> pr1.update(pr2)
            >>> pr1
            {'a': 1, 'b': 2}
            >>> pr1.no_grad_parameters
            {'b'}
        """
        _check_pr_type(others)
        super().update(others)
        conflict = (self.no_grad_parameters & others.requires_grad_parameters
                    ) | (others.no_grad_parameters
                         & self.requires_grad_parameters)
        if conflict:
            raise ValueError(
                "Parameter conflict, {} require grad in some parameter \
resolver and not require grad in other parameter resolver ".format(conflict))
        self.no_grad_parameters.update(others.no_grad_parameters)
        self.requires_grad_parameters.update(others.requires_grad_parameters)

    def mindspore_data(self):
        """
        Generate data for PQC operator.

        Returns:
            Dict.
        """
        m_data = {
            'gate_params_names': [],
            'gate_coeff': [],
            'gate_requires_grad': []
        }
        for k, v in self.items():
            m_data['gate_params_names'].append(k)
            m_data['gate_coeff'].append(float(v))
            m_data['gate_requires_grad'].append(
                k in self.requires_grad_parameters)
        return m_data

    def expression(self):
        res = 0
        for k, v in self.items():
            res += sp.Symbol(k) * v
        return res

    def __str__(self):
        return str(self.expression())

    def __repr__(self):
        return str(self.expression())


def _check_pr_type(pr):
    if not isinstance(pr, ParameterResolver):
        raise TypeError("Require a ParameterResolver, but get {}".format(
            type(pr)))


_num_type = (int, float, np.int32, np.int64, np.float32, np.float64)
