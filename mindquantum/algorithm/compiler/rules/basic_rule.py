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

from mindquantum.core.circuit import Circuit

from ..dag import DAGCircuit
from .compiler_logger import CompileLog as CLog
from .compiler_logger import LogIndentation


# pylint: disable=too-few-public-methods,invalid-name
class BasicCompilerRule(ABC):
    """
    The basic compiler rule class.

    Compiler rule will handle a :class:`~.algorithm.compiler.DAGCircuit` and
    compile it following the rule in the ``do`` method. You must implement
    the ``do`` method when inheritance a child compiler rule. Please make sure
    that the ``do`` method will return a bool, which will tell whether the compiler
    successful compiled the circuit.

    Args:
        rule_name (str): The name of this compiler rule.
        log_level (int): The display log level. If ``0``, log will be disabled.
            If ``1``, a brief log will be displayed. If ``2``, a detail log
            will be displayed. Default: ``0``.
    """

    def __init__(self, rule_name="BasicCompilerRule", log_level=0):
        """Initialize a basic compiler rule."""
        self.rule_name = rule_name
        self.log_level = log_level

    def __repr__(self):
        """Get string expression of rule."""
        return f"{self.rule_name}<>"

    def set_log_level(self, log_level: int):
        """
        Set display log level.

        Args:
            log_level (int): the display log level. Could be ``0``, ``1`` or ``2``. For more explanation of log level,
                please refers to :class:`~.algorithm.compiler.BasicCompilerRule`.
        """
        self.log_level = log_level
        return self

    @abstractmethod
    def do(self, dag_circuit: DAGCircuit) -> bool:
        """
        Inplace applying this compiler rule to the :class:`~.algorithm.compiler.DAGCircuit`.

        Args:
            dag_circuit (:class:`~.algorithm.compiler.DAGCircuit`): The DAG graph of quantum circuit.
        """


class SequentialCompiler(BasicCompilerRule):
    """
    A sequential of compiler.

    Every compiler rule in this sequence will be execute one by one.

    Args:
        compilers (List[:class:`~.algorithm.compiler.BasicCompilerRule`]): all compiler rules you want.
        rule_name (str): name of this compiler. Default: ``"SequentialCompiler"``.
        log_level (int): the display log level. Could be ``0``, ``1`` or ``2``. For more explanation of log level,
                please refers to :class:`~.algorithm.compiler.BasicCompilerRule`. Default: ``0``.
    """

    def __init__(self, compilers: typing.List[BasicCompilerRule], rule_name="SequentialCompiler", log_level=0):
        """Initialize a sequential compiler rule."""
        self.compilers = compilers
        super().__init__(rule_name, log_level)

    def __repr__(self):
        """Get string expression of rule."""
        strs = [f'{self.rule_name}<']
        for compiler in self.compilers:
            for i in compiler.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append('>')
        return '\n'.join(strs)

    def set_all_log_level(self, log_level: int):
        """
        Set log level for all compiler rule in this sequence.

        Args:
            log_level (int): the display log level. Could be ``0``, ``1`` or ``2``. For more explanation of log level,
                please refers to :class:`~.algorithm.compiler.BasicCompilerRule`.
        """
        self.log_level = log_level
        for compiler in self.compilers:
            if hasattr(compiler, "set_all_log_level"):
                compiler.set_all_log_level(log_level)
            else:
                compiler.set_log_level(log_level)
        return self

    def do(self, dag_circuit: DAGCircuit) -> bool:
        """
        Apply sequential compiler to dag circuit.

        Args:
            dag_circuit (:class:`~.algorithm.compiler.DAGCircuit`): The DAG graph of quantum circuit.
        """
        compiled = False
        child_name = ', '.join(CLog.R2(compiler.rule_name) for compiler in self.compilers)
        CLog.log(f"Running {CLog.R1(self.rule_name)}: {len(self.compilers)} child ({child_name}, ).", 1, self.log_level)
        with LogIndentation() as _:
            states = [compiler.do(dag_circuit) for compiler in self.compilers]
            CLog.log(f"{CLog.R1(self.rule_name)}: state for each rule -> {CLog.ShowState(states)}", 2, self.log_level)
        compiled = any(states)
        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfully compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled


class KroneckerSeqCompiler(SequentialCompiler):
    """
    Kronecker sequential compiler.

    Every compiler rule in this KroneckerSeqCompiler will be execute until they do not compile any gate.

    Args:
        compilers (List[:class:`~.algorithm.compiler.BasicCompilerRule`]): compiler rules.
        rule_name(str): the name of compiler rule. Default: ``"KroneckerSeqCompiler"``.
        log_level (int): the display log level. Could be ``0``, ``1`` or ``2``. Default: ``0``.
            For more explanation of log level, please refers to :class:`~.algorithm.compiler.BasicCompilerRule`.
    """

    def __init__(self, compilers: typing.List[BasicCompilerRule], rule_name="KroneckerSeqCompiler", log_level=0):
        """Initialize a KroneckerSeqCompiler."""
        super().__init__(compilers, rule_name, log_level)

    def do(self, dag_circuit: DAGCircuit) -> bool:
        """
        Apply kronecker compiler to dag circuit.

        Args:
            dag_circuit (:class:`~.algorithm.compiler.DAGCircuit`): The DAG graph of quantum circuit.
        """
        compiled = False
        child_name = ', '.join(CLog.R2(compiler.rule_name) for compiler in self.compilers)
        CLog.log(f"Running {CLog.R1(self.rule_name)}: {len(self.compilers)} child ({child_name}, ).", 1, self.log_level)
        with LogIndentation() as _:
            while True:
                states = [compiler.do(dag_circuit) for compiler in self.compilers]
                CLog.log(
                    f"{CLog.R1(self.rule_name)}: state for each rule -> {CLog.ShowState(states)}", 2, self.log_level
                )
                if any(states):
                    compiled = True
                else:
                    break
        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfully compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled


def compile_circuit(compiler: BasicCompilerRule, circ: Circuit) -> Circuit:
    """
    Directly compile a given circuit and return the result.

    Args:
        compiler (:class:`~.algorithm.compiler.BasicCompilerRule`): compile rules.
        circ (:class:`~.core.circuit.Circuit`): the input quantum circuit.

    Returns:
        :class:`~.core.circuit.Circuit`, the final circuit compiled by given compiler.

    Examples:
        >>> from mindquantum.algorithm.compiler import compile_circuit, BasicDecompose
        >>> from mindquantum.core import gates as G
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit([G.Rxx('a').on([0, 1]), G.X(0, [1, 2])])
        >>> circ
              ┏━━━━━━━━┓ ┏━━━┓
        q0: ──┨        ┠─┨╺╋╸┠───
              ┃        ┃ ┗━┳━┛
              ┃ Rxx(a) ┃   ┃
        q1: ──┨        ┠───■─────
              ┗━━━━━━━━┛   ┃
                           ┃
        q2: ───────────────■─────
        >>> compile_circuit(BasicDecompose(), circ)
              ┏━━━┓                       ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━━┓ ┏━━━┓
        q0: ──┨ H ┠───■───────────────■───┨ H ┠─┨ H ┠─┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠─┨ T ┠─┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠─↯─
              ┗━━━┛   ┃               ┃   ┗━━━┛ ┗━━━┛ ┗━┳━┛ ┗━━━━┛ ┗━┳━┛ ┗━━━┛ ┗━┳━┛ ┗━━━━┛ ┗━┳━┛
              ┏━━━┓ ┏━┻━┓ ┏━━━━━━━┓ ┏━┻━┓ ┏━━━┓         ┃            ┃           ┃            ┃
        q1: ──┨ H ┠─┨╺╋╸┠─┨ RZ(a) ┠─┨╺╋╸┠─┨ H ┠─────────■────────────╂───────────■────────────╂───↯─
              ┗━━━┛ ┗━━━┛ ┗━━━━━━━┛ ┗━━━┛ ┗━━━┛                      ┃                        ┃
                                                                     ┃                        ┃
        q2: ─────────────────────────────────────────────────────────■────────────────────────■───↯─
              ┏━━━┓ ┏━━━┓
        q0: ──┨ T ┠─┨ H ┠────────────────
              ┗━━━┛ ┗━━━┛
              ┏━━━┓ ┏━━━┓ ┏━━━━┓ ┏━━━┓
        q1: ──┨ T ┠─┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠───
              ┗━━━┛ ┗━┳━┛ ┗━━━━┛ ┗━┳━┛
                      ┃   ┏━━━┓    ┃
        q2: ──────────■───┨ T ┠────■─────
                          ┗━━━┛
    """
    dag_circ = DAGCircuit(circ)
    compiler.do(dag_circ)
    return dag_circ.to_circuit()
