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
"""Neighbor canceler compiler rule."""
from mindquantum.utils.type_value_check import _check_input_type

from ..dag import DAGCircuit, GateNode, try_merge
from .basic_rule import BasicCompilerRule, KroneckerSeqCompiler
from .compiler_logger import CompileLog as CLog
from .compiler_logger import LogIndentation


# pylint: disable=too-many-nested-blocks,too-many-branches,too-few-public-methods
class SimpleNeighborCanceler(BasicCompilerRule):
    """Merge two nearby gate if possible."""

    def __init__(self):
        """Initialize a neighbor canceler compiler rule."""
        super().__init__("SimpleNeighborCanceler")

    def _canceler(self, current_node, fc_pair_consided, dag_circuit):
        """Merge two gate."""
        compiled = False
        for local in current_node.local:
            if not current_node.child:
                continue
            child_node = current_node.child[local]
            fc_pair = (current_node, child_node)
            if fc_pair in fc_pair_consided:
                continue
            fc_pair_consided.add(fc_pair)
            if not isinstance(child_node, GateNode):
                continue
            if isinstance(current_node, GateNode):
                compiled = self._merge_two_gates(current_node, child_node, fc_pair_consided, dag_circuit) or compiled
            compiled = self._canceler(child_node, fc_pair_consided, dag_circuit) or compiled
        return compiled

    def do(self, dag_circuit: DAGCircuit) -> bool:
        """
        Apply neighbor canceler compiler rule.

        Args:
            dag_circuit (:class:`~.algorithm.compiler.DAGCircuit`): The DAG graph of quantum circuit.
        """
        _check_input_type("dag_circuit", DAGCircuit, dag_circuit)

        fc_pair_consided = set()
        compiled = False
        CLog.log(f"Running {CLog.R1(self.rule_name)}.", 1, self.log_level)
        with LogIndentation() as _:
            for current_node in dag_circuit.head_node.values():
                compiled = self._canceler(current_node, fc_pair_consided, dag_circuit) or compiled
        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfully compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled

    def _merge_two_gates(self, current_node: GateNode, child_node: GateNode, fc_pair_consided, dag_circuit):
        """Merge two gates."""
        compiled = False
        state, res, global_phase = try_merge(current_node, child_node)
        if not state:
            return False
        compiled = True
        with LogIndentation() as _:
            CLog.log(
                f"{CLog.R1(self.rule_name)}: merge {CLog.B(current_node.gate)} and {CLog.B(child_node.gate)}.",
                2,
                self.log_level,
            )
        if global_phase:
            dag_circuit.global_phase.coeff += global_phase.coeff
        for node in res:
            compiled = self._canceler(node, fc_pair_consided, dag_circuit) or compiled
        return compiled


class FullyNeighborCanceler(KroneckerSeqCompiler):
    """Merge neighbor gate until we cannot merge anymore gates."""

    def __init__(self):
        """Initialize fully neighbor canceler compile rule."""
        rule_set = [SimpleNeighborCanceler()]
        super().__init__(rule_set)
        self.rule_name = "FullyNeighborCanceler"
