# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Parameter Generator."""

import mindquantum as mq

from .parameterresolver import ParameterResolver


class PRGenerator:
    """
    Generate parameters one by one.

    Args:
        name (str): The main name of parameters. Default: ``'p'``.
        prefix (str): The prefix of parameters. Default: ``''``.
        suffix (str): The suffix of parameters. Default: ``''``.
        dtype (mindquantum.dtype): the data type of this parameter resolver. If ``None``,
            dtype would be ``mindquantum.float64``. Default: ``None``.

    Examples:
        >>> from mindquantum.core.parameterresolver import PRGenerator
        >>> pr_gen = PRGenerator()
        >>> print(pr_gen.new())
        p0
        >>> print(pr_gen.new(suffix='a'))
        p1_a
        >>> pr_gen.reset()
        >>> print(pr_gen.new())
        p0
        >>> pr_gen.size()
        1
    """

    def __init__(self, name='p', prefix: str = '', suffix: str = '', dtype=None):
        """Initialize a pr generator."""
        if dtype is None:
            self.dtype = mq.float64
        else:
            self.dtype = dtype
        self.name = name
        self.prefix = prefix
        self.suffix = suffix
        if prefix:
            self.prefix += '_'
        if suffix:
            self.suffix = '_' + self.suffix
        self.current_idx = 0
        self.all_pr = []

    def reset(self):
        """Reset the pr generator to initialize state."""
        self.current_idx = 0
        self.all_pr = []

    def new(self, prefix: str = '', suffix: str = '') -> ParameterResolver:
        """
        Generate a new parameter.

        Args:
            prefix (str): The extra prefix when generate this new parameter. Default: ``''``.
            suffix (str): The extra suffix when generate this new parameter. Default: ``''``.

        Examples:
            >>> from mindquantum.core.parameterresolver import PRGenerator
            >>> pr_gen = PRGenerator(prefix='l')
            >>> print(pr_gen.new(suffix='a'))
            l_p1_a
        """
        if prefix:
            prefix += '_'
        if suffix:
            suffix = '_' + suffix
        out = ParameterResolver(
            f'{prefix}{self.prefix}{self.name}{self.current_idx}{self.suffix}{suffix}', dtype=self.dtype
        )
        self.all_pr.append(out)
        self.current_idx += 1
        return out

    def size(self):
        """Get the total size of parameters that generated."""
        return self.current_idx
