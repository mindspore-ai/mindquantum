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
"""SABRE algorithm to implement qubit mapping."""
import typing

from ...core.circuit import Circuit
from ...core.gates import SWAP
from ...device import QubitsTopology
from ...mqbackend.device import SABRE as SABRE_  # pylint: disable=import-error


# pylint: disable=too-few-public-methods
class SABRE:
    """
    SABRE algorithm to implement qubit mapping.

    For more detail of SABRE algorithm, please refer to
    `Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices <https://arxiv.org/abs/1809.02573>`_.

    Args:
        circuit (:class:`~.core.circuit.Circuit`): The quantum circuit you need to do qubit mapping. Currently we only
            support circuit constructed by one or two qubits gate, control qubit included.
        topology (:class:`~.device.QubitsTopology`): The hardware qubit topology. Currently we only support
            connected coupling graph.
    """

    def __init__(self, circuit: Circuit, topology: QubitsTopology):
        """Initialize a sabre qubit mapping solver."""
        self.circuit = circuit
        self.topology = topology
        self.cpp_solver = SABRE_(self.circuit.get_cpp_obj(), self.topology.__get_cpp_obj__())

        def check_connected(topology: QubitsTopology) -> bool:
            """Check whether topology graph is connected."""
            qids = topology.all_qubit_id()
            if not qids:
                return False
            edges = topology.edges_with_id()
            graph = {qid: [] for qid in qids}
            for (x, y) in edges:
                graph[x].append(y)
                graph[y].append(x)

            vis = {qid: False for qid in qids}

            def dfs(x: int):
                vis[x] = True
                for y in graph[x]:
                    if not vis[y]:
                        dfs(y)

            dfs(qids.pop())
            return all(vis.values())

        if not check_connected(topology):
            raise ValueError(
                'The current mapping algorithm SABRE only supports connected graphs, '
                'please manually assign some lines to connected subgraphs.'
            )

    def solve(
        self, iter_num: int, w: float, delta1: float, delta2: float
    ) -> typing.Union[Circuit, typing.List[int], typing.List[int]]:
        """
        Solve qubit mapping problem with SABRE algorithm.

        Args:
            iter_num (int): The iteration number for when solving mapping problem.
            w (float): The w parameter. For more detail, please refers to the paper.
            delta1 (float): The delta1 parameter. For more detail, please refers to the paper.
            delta2 (float): The delta2 parameter. For more detail, please refers to the paper.

        Returns:
            Tuple[:class:`~.core.circuit.Circuit`, List[int], List[int]], a quantum
                circuit that can execute on given device, the initial mapping order,
                and the final mapping order.
        """
        gate_info, (init_map, final_map) = self.cpp_solver.solve(iter_num, w, delta1, delta2)
        new_circ = Circuit()
        for idx, p1, p2 in gate_info:
            if idx == -1:
                new_circ += SWAP.on([p1, p2])
            else:
                ori_gate = self.circuit[idx]
                if p1 == p2:
                    new_circ += ori_gate.on(p1)
                else:
                    if len(ori_gate.obj_qubits) == 1:
                        new_circ += ori_gate.on(p1, p2)
                    else:
                        new_circ += ori_gate.on([p1, p2])
        return new_circ, init_map, final_map
