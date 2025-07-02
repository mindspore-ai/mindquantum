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

"""MQSABRE algorithm to implement qubit mapping."""
from typing import List, Tuple

from ...core.circuit import Circuit
from ...core.gates import SWAP
from ...device import QubitsTopology
from ...mqbackend.device import MQ_SABRE as MQ_SABRE_  # pylint: disable=import-error


# pylint: disable=too-few-public-methods
class MQSABRE:
    """
    MQSABRE algorithm for hardware-aware qubit mapping optimization.

    MQSABRE extends the SABRE (SWAP-based BidiREctional heuristic search) algorithm by incorporating
    hardware-specific characteristics into the mapping optimization process. The algorithm performs
    initial mapping and routing optimization in three phases:

    1. Initial mapping: Uses a graph-center-based approach to generate an initial mapping that
       minimizes the average distance between frequently interacting qubits.
    2. Mapping optimization: Employs bidirectional heuristic search with a hardware-aware cost function.
    3. Circuit transformation: Inserts SWAP gates and transforms the circuit to be compatible with
       hardware constraints.

    The algorithm uses a weighted cost function that combines three metrics:
    H = α₁D + α₂K + α₃T
    where:

    - D: Shortest path distance between qubits in the coupling graph
    - K: Error rate metric derived from CNOT and SWAP success rates
    - T: Gate execution time metric considering CNOT and SWAP durations
    - α₁, α₂, α₃: Weight parameters for balancing different optimization objectives

    Args:
        circuit (:class:`~.core.circuit.Circuit`): The quantum circuit to be mapped. Currently only supports
            circuits composed of single-qubit gates and two-qubit gates (including controlled gates).
        topology (:class:`~.device.QubitsTopology`): The hardware qubit topology. Must be a connected
            coupling graph.
        cnoterrorandlength (List[Tuple[Tuple[int, int], List[float]]]): Hardware-specific CNOT characteristics.
            Each entry contains a tuple (i, j) specifying the physical qubit pair in the topology, and a list
            [error_rate, gate_time] where error_rate is the CNOT error rate between qubits i and j (range: [0, 1]),
            and gate_time is the CNOT execution time in arbitrary units.

    Raises:
        ValueError: If the topology is not a connected graph.

    Examples:
        >>> from mindquantum.algorithm.mapping import MQSABRE
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import RX, X
        >>> from mindquantum.device import GridQubits
        >>> # Create a quantum circuit
        >>> circ = Circuit()
        >>> circ += RX('a').on(0)
        >>> circ += RX('b').on(1)
        >>> circ += X.on(1, 0)
        >>> # Define hardware characteristics
        >>> cnot_data = [
        ...     ((0, 1), [0.001, 250.0]),  # CNOT 0->1: 0.1% error, 250ns
        ...     ((1, 2), [0.002, 300.0]),  # CNOT 1->2: 0.2% error, 300ns
        ... ]
        >>> # Create a linear topology: 0-1-2
        >>> topology = GridQubits(1, 3)
        >>> # Initialize and run MQSABRE
        >>> solver = MQSABRE(circ, topology, cnot_data)
        >>> new_circ, init_map, final_map = solver.solve()
    """

    def __init__(
        self, circuit: Circuit, topology: QubitsTopology, cnoterrorandlength: List[Tuple[Tuple[int, int], List[float]]]
    ):
        """Initialize a sabre qubit mapping solver."""
        self.circuit = circuit.remove_barrier()
        self.topology = topology
        self.cnoterrorandlength = cnoterrorandlength
        self.cpp_solver = MQ_SABRE_(
            self.circuit.get_cpp_obj(), self.topology.__get_cpp_obj__(), self.cnoterrorandlength
        )

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
                'The current mapping algorithm MQSABRE only supports connected graphs, '
                'please manually assign some lines to connected subgraphs.'
            )

    def solve(
        self, w: float = 0.5, alpha1: float = 0.3, alpha2: float = 0.2, alpha3: float = 0.1
    ) -> Tuple[Circuit, List[int], List[int]]:
        """
        Solve the qubit mapping problem using the MQSABRE algorithm.

        The method performs three main steps:
        1. Constructs the distance matrix D using Floyd-Warshall algorithm
        2. Computes hardware-specific matrices K (error rates) and T (gate times)
        3. Performs heuristic search to optimize the mapping while considering the combined cost function

        Args:
            w (float, optional): Look-ahead weight parameter that balances between current and future
                gate operations in the heuristic search. Range: [0, 1].
                When w > 0.5, it favors future operations, potentially reducing circuit depth.
                When w < 0.5, it prioritizes current operations, potentially reducing total gate count.
                Defaults: 0.5.
            alpha1 (float, optional): Weight for the distance metric (D) in the cost function.
                Higher values prioritize minimizing qubit distances. Defaults: 0.3.
            alpha2 (float, optional): Weight for the error rate metric (K).
                Higher values prioritize connections with lower error rates. Defaults: 0.2.
            alpha3 (float, optional): Weight for the gate time metric (T).
                Higher values prioritize faster gate execution paths. Defaults: 0.1.

        Returns:
            - mapped_circuit (:class:`~.core.circuit.Circuit`), The transformed circuit with inserted SWAP gates
            - initial_mapping (List[int]), Initial mapping from logical to physical qubits
            - final_mapping (List[int]), Final mapping from logical to physical qubits

        Examples:
            >>> # Use default parameters
            >>> new_circ, init_map, final_map = solver.solve()
            >>> # Prioritize error rate optimization
            >>> new_circ, init_map, final_map = solver.solve(alpha2=0.5)
            >>> # Focus on circuit depth reduction
            >>> new_circ, init_map, final_map = solver.solve(w=0.7)
        """
        gate_info, (init_map, final_map) = self.cpp_solver.solve(w, alpha1, alpha2, alpha3)
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
