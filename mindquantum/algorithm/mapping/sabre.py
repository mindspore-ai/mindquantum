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
    SABRE (SWAP-based BidiREctional heuristic search) algorithm for qubit mapping optimization.

    Due to physical constraints in real quantum hardware, not all qubits can directly interact with each other.
    SABRE algorithm enables arbitrary quantum circuits to run on specific quantum hardware topologies by inserting
    SWAP gates and remapping qubits. It employs a bidirectional heuristic search method that minimizes a cost
    function considering both current and future gate operations to find optimal mapping solutions.

    Reference:
        Gushu Li, Yufei Ding, Yuan Xie: "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices",
        ASPLOS 2019. https://arxiv.org/abs/1809.02573

    Args:
        circuit (:class:`~.core.circuit.Circuit`): The quantum circuit to be mapped. Currently only supports
            circuits composed of single-qubit gates and two-qubit gates (including controlled gates).
        topology (:class:`~.device.QubitsTopology`): The hardware qubit topology. Currently only supports
            connected coupling graphs.

    Examples:
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import RX, X
        >>> from mindquantum.device import GridQubits
        >>> from mindquantum.algorithm.mapping import SABRE
        >>> # Create a simple quantum circuit
        >>> circ = Circuit()
        >>> circ += RX('a').on(0)
        >>> circ += RX('b').on(1)
        >>> circ += RX('c').on(2)
        >>> circ += X.on(1, 0)
        >>> circ += X.on(2, 1)
        >>> circ += X.on(0, 2)
        >>> # Create a 2x2 grid topology
        >>> topo = GridQubits(2, 2)
        >>> # Use SABRE for mapping
        >>> solver = SABRE(circ, topo)
        >>> new_circ, init_map, final_map = solver.solve()
    """

    def __init__(self, circuit: Circuit, topology: QubitsTopology):
        """Initialize a sabre qubit mapping solver."""
        self.circuit = circuit.remove_barrier()
        self.topology = topology
        self.cpp_solver = SABRE_(self.circuit.get_cpp_obj(), self.topology.__get_cpp_obj__())

        def check_connected(topology: QubitsTopology) -> bool:
            """Check whether topology graph is connected."""
            qids = topology.all_qubit_id()
            if not qids:
                return False
            edges = topology.edges_with_id()
            graph = {qid: [] for qid in qids}
            for x, y in edges:
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
        self, iter_num: int = 5, w: float = 0.5, delta1: float = 0.3, delta2: float = 0.2
    ) -> typing.Union[Circuit, typing.List[int], typing.List[int]]:
        """
        Solve the qubit mapping problem using the SABRE algorithm.

        The method employs bidirectional heuristic search to find optimal qubit mapping solutions.
        Key steps include:
        1. Generate random initial mapping
        2. Execute forward-backward-forward traversal to optimize initial mapping
        3. Perform final forward traversal with optimized mapping to generate physical circuit with SWAP gates

        Args:
            iter_num (int, optional): Number of forward-backward-forward traversal iterations. Each iteration
                starts with a different initial mapping. Default: 5.
            w (float, optional): Weight parameter in cost function H = H_current + w * H_future.
                Larger w (>0.5) favors future operations, potentially reducing circuit depth;
                Smaller w (<0.5) favors current operations, potentially reducing total gate count.
                Default: 0.5.
            delta1 (float, optional): Decay parameter for single-qubit gates. Affects how algorithm updates
                decay values after single-qubit operations. Default: 0.3.
            delta2 (float, optional): Decay parameter for two-qubit gates (CNOT, SWAP). Controls how algorithm
                distributes SWAP operations in space and time. Since one SWAP equals three CNOTs, SWAP operations
                add 3*delta2 to decay values. Default: 0.2.

        Returns:
            - mapped_circuit (:class:`~.core.circuit.Circuit`), Quantum circuit compatible with hardware
              topology after adding SWAP gates
            - initial_mapping (List[int]), Mapping from logical to physical qubits at the start of execution
            - final_mapping (List[int]), Mapping from logical to physical qubits at the end of execution

        Examples:
            >>> # Use default parameters
            >>> new_circ, init_map, final_map = solver.solve()
            >>> # Or customize parameters
            >>> new_circ, init_map, final_map = solver.solve(iter_num=10, w=0.7)
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
