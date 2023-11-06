"""MQSABRE algorithm to implement qubit mapping."""
import typing
from typing import List, Tuple

from ...core.circuit import Circuit
from ...core.gates import SWAP
from ...device import QubitsTopology
from ...mqbackend.device import MQ_SABRE as MQ_SABRE_  # pylint: disable=import-error


# pylint: disable=too-few-public-methods
class MQSABRE:
    """
    MQSABRE algorithm to implement qubit mapping.

    Args:
        circuit (:class:`~.core.circuit.Circuit`): The quantum circuit you need to do qubit mapping. Currently we only
            support circuit constructed by one or two qubits gate, control qubit included.
        topology (:class:`~.device.QubitsTopology`): The hardware qubit topology. Currently we only support
            connected coupling graph.
        cnoterrorandlength (:the property info of CNOT gate)
    """

    def __init__(
        self, circuit: Circuit, topology: QubitsTopology, cnoterrorandlength: List[Tuple[Tuple[int, int], List[float]]]
    ):
        """Initialize a sabre qubit mapping solver."""
        self.circuit = circuit
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
        self, w: float, alpha1: float, alpha2: float, alpha3: float
    ) -> typing.Union[Circuit, typing.List[int], typing.List[int]]:
        """
        Solve qubit mapping problem with SABRE algorithm.

        Args:
            w (float): The w parameter. For more detail, please refers to the paper.
            delta1 (float): The delta1 parameter. For more detail, please refers to the paper.
            delta2 (float): The delta2 parameter. For more detail, please refers to the paper.

        Returns:
            Tuple[:class:`~.core.circuit.Circuit`, List[int], List[int]], a quantum
                circuit that can execute on given device, the initial mapping order,
                and the final mapping order.
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
