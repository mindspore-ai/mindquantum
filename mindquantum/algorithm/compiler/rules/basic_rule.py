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

from .compiler_logger import CompileLog as CLog, LogIndentation
from ..dag import DAGCircuit
from mindquantum.core.circuit import Circuit

# pylint: disable=too-few-public-methods,invalid-name
class BasicCompilerRule(ABC):
    """The basic compiler rule class."""

    def __init__(self, rule_name="BasicCompilerRule", log_level=0):
        """Initialize a basic compiler rule."""
        self.rule_name = rule_name
        self.log_level = log_level

    def set_log_level(self, log_level: int):
        self.log_level = log_level
        return self

    def __repr__(self):
        """String expression of rule."""
        return f"{self.rule_name}<>"

    @abstractmethod
    def do(self, dagcircuit) -> bool:
        """Apply this compiler rule."""


class SequentialCompiler(BasicCompilerRule):
    """A sequential of compiler."""

    def __init__(self, compilers: typing.List[BasicCompilerRule], rule_name="SequentialCompiler", log_level=0):
        self.compilers = compilers
        super().__init__(rule_name, log_level)

    def __repr__(self):
        """String expression of rule."""
        strs = [f'{self.rule_name}<']
        for compiler in self.compilers:
            for i in compiler.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append('>')
        return '\n'.join(strs)

    def set_all_log_level(self, log_level):
        self.log_level = log_level
        for compiler in self.compilers:
            if hasattr(compiler, "set_all_log_level"):
                compiler.set_all_log_level(log_level)
            else:
                compiler.set_log_level(log_level)
        return self

    def do(self, dagcircuit):
        """Apply sequential compiler to dag circuit."""
        compiled = False
        child_name = ', '.join(CLog.R2(compiler.rule_name) for compiler in self.compilers)
        CLog.log(f"Running {CLog.R1(self.rule_name)}: {len(self.compilers)} child ({child_name}, ).", 1, self.log_level)
        with LogIndentation() as _:
            states = [compiler.do(dagcircuit) for compiler in self.compilers]
            CLog.log(f"{CLog.R1(self.rule_name)}: state for each rule -> {CLog.ShowState(states)}", 2, self.log_level)
        compiled = any(states)
        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfule compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled


class KroneckerSeqCompiler(SequentialCompiler):
    """Kronecher sequential compiler."""

    def __init__(self, compilers: typing.List[BasicCompilerRule], rule_name="KroneckerSeqCompiler", log_level=0):
        super().__init__(compilers, rule_name, log_level)

    def do(self, dagcircuit):
        compiled = False
        child_name = ', '.join(CLog.R2(compiler.rule_name) for compiler in self.compilers)
        CLog.log(f"Running {CLog.R1(self.rule_name)}: {len(self.compilers)} child ({child_name}, ).", 1, self.log_level)
        with LogIndentation() as _:
            while True:
                states = [compiler.do(dagcircuit) for compiler in self.compilers]
                CLog.log(f"{CLog.R1(self.rule_name)}: state for each rule -> {CLog.ShowState(states)}", 2, self.log_level)
                if any(states):
                    compiled = True
                else:
                    break
        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfule compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled

def compile_circuit(compiler:BasicCompilerRule, circ:Circuit)->Circuit:
    dag_circ = DAGCircuit(circ)
    compiler.do(dag_circ)
    return dag_circ.to_circuit()
