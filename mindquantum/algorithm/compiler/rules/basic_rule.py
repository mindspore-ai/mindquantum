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
"""Basic compiler rule."""
import typing
from abc import ABC, abstractmethod


# pylint: disable=too-few-public-methods,invalid-name
class BasicCompilerRule(ABC):
    """The basic compiler rule class."""

    def __init__(self, rule_name):
        """Initialize a basic compiler rule."""
        self.rule_name = rule_name

    def __repr__(self):
        """String expression of rule."""
        return "BasicCompilerRule<>"

    @abstractmethod
    def do(self, dagcircuit):
        """Apply this compiler rule."""


class SequentialCompiler(BasicCompilerRule):
    """A sequential of compiler."""

    def __init__(self, compilers: typing.List[BasicCompilerRule]):
        self.compilers = compilers
        super().__init__("SequentialCompiler")

    def __repr__(self):
        """String expression of rule."""
        strs = ['SequentialCompiler<']
        for compiler in self.compilers:
            for i in compiler.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append('>')
        return '\n'.join(strs)

    def do(self, dagcircuit):
        """Apply sequential compiler to dag circuit."""
        for compiler in self.compilers:
            compiler.do(dagcircuit)
