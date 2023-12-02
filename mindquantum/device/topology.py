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
"""Qubit node an topology of quantum chip."""

import copy
import numbers
import sys
import typing
from typing import Optional

# pylint: disable=wrong-import-position
if sys.version_info < (3, 8):
    from typing_extensions import Literal, get_args
else:
    from typing import Literal, get_args

from rich.console import Console

from mindquantum.utils.type_value_check import _check_input_type, _check_int_type

from ..mqbackend.device import QubitNode as QubitNode_
from ..mqbackend.device import QubitsTopology as QubitsTopology_

AVA_SHOW_METHOD = Literal['mpl', 'svg']  # pylint: disable=invalid-name


class QubitNode:
    """
    Qubit node.

    A qubit node has a id, a position and a color (if you want to draw it). You can connect two qubits with
    '>>' or '<<' and disconnect two qubits with '<' or '>'.

    Args:
        qubit_id (int): the identity number of this qubit.
        color (str): the face color of this qubit.
        poi_x (float): x position in canvas.
        poi_y (float): y position in canvas.

    Examples:
        >>> from mindquantum.device import QubitNode
        >>> q0 = QubitNode(0)
        >>> q1 = QubitNode(1)
        >>> q = q0 << q1
        >>> q0.qubit_id == q.qubit_id
        True
    """

    def __init__(self, qubit_id: int, color: str = '#000000', poi_x: float = 0.0, poi_y: float = 0.0) -> None:
        """Initialize a qubit node."""
        _check_int_type("qubit_id", qubit_id)
        _check_input_type("color", str, color)
        _check_input_type("poi_x", numbers.Real, poi_x)
        _check_input_type("poi_y", numbers.Real, poi_y)
        self.id_ = qubit_id
        self.color_ = color
        self.poi_x_ = poi_x
        self.poi_y_ = poi_y
        self.neighbor: typing.Set[int] = set()

    def __get_cpp_obj__(self):
        """Get cpp obj."""
        return QubitNode_(self.id_, self.color_, self.poi_x_, self.poi_y_, self.neighbor)

    def __gt__(self, other: "QubitNode") -> "QubitNode":
        """
        Disconnect with other qubit node and return rhs.

        Args:
            other (:class:`~.device.QubitNode`): qubit node you want to disconnect with.

        Returns:
            :class:`~.device.QubitNode`, the right hand size qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> from mindquantum.device import QubitsTopology
            >>> q0 = QubitNode(0)
            >>> q1 = QubitNode(1)
            >>> topology = QubitsTopology([q0, q1])
            >>> q0 << q1
            >>> topology.is_coupled_with(0, 1)
            True
            >>> q0 > q1
            >>> topology.is_coupled_with(0, 1)
            False
        """
        if self.qubit_id == other.qubit_id:
            raise RuntimeError("Cannot disconnect itself.")
        if other.qubit_id in self.neighbor:
            self.neighbor.remove(other.qubit_id)
        if self.qubit_id in other.neighbor:
            other.neighbor.remove(self.qubit_id)

        return other

    def __int__(self) -> int:
        """
        Convert the qubit to int.

        Returns:
            int, the id of qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> q0 = QubitNode(0)
            >>> int(q0)
            0
        """
        return self.qubit_id

    def __lshift__(self, other: "QubitNode") -> "QubitNode":
        """
        Connect with other qubit node and return lhs.

        Args:
            other (:class:`~.device.QubitNode`): qubit node you want to connect with.

        Returns:
            :class:`~.device.QubitNode`, the left hand size qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> from mindquantum.device import QubitsTopology
            >>> q0 = QubitNode(0)
            >>> q1 = QubitNode(1)
            >>> topology = QubitsTopology([q0, q1])
            >>> q = q0 << q1
            >>> q.qubit_id == q0.qubit_id
            True
            >>> topology.is_coupled_with(0, 1)
            True
        """
        _check_input_type("other", QubitNode, other)
        if self.qubit_id == other.qubit_id:
            raise RuntimeError("Cannot disconnect itself.")
        self.neighbor.add(other.qubit_id)
        other.neighbor.add(self.qubit_id)

        return self

    def __lt__(self, other: "QubitNode") -> "QubitNode":
        """
        Disconnect with other qubit node and return lhs.

        Args:
            other (:class:`~.device.QubitNode`): qubit node you want to connect with.

        Returns:
            :class:`~.device.QubitNode`, the left hand size qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> from mindquantum.device import QubitsTopology
            >>> q0 = QubitNode(0)
            >>> q1 = QubitNode(1)
            >>> topology = QubitsTopology([q0, q1])
            >>> q0 << q1
            >>> topology.is_coupled_with(0, 1)
            True
            >>> q = q0 < q1
            >>> topology.is_coupled_with(0, 1)
            False
            >>> q.qubit_id == q0.qubit_id
            True
        """
        _check_input_type("other", QubitNode, other)
        if self.qubit_id == other.qubit_id:
            raise RuntimeError("Cannot disconnect itself.")
        if other.qubit_id in self.neighbor:
            self.neighbor.remove(other.qubit_id)
        if self.qubit_id in other.neighbor:
            other.neighbor.remove(self.qubit_id)

        return self

    def __rshift__(self, other: "QubitNode") -> "QubitNode":
        """
        Connect with other qubit node and return rhs.

        Args:
            other (:class:`~.device.QubitNode`): qubit node you want to connect with.

        Returns:
            :class:`~.device.QubitNode`, the right hand side qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> from mindquantum.device import QubitsTopology
            >>> q0 = QubitNode(0)
            >>> q1 = QubitNode(1)
            >>> topology = QubitsTopology([q0, q1])
            >>> q = q0 >> q1
            >>> q.qubit_id == q1.qubit_id
            True
            >>> topology.is_coupled_with(0, 1)
            True
        """
        _check_input_type("other", QubitNode, other)
        if self.qubit_id == other.qubit_id:
            raise RuntimeError("Cannot connect itself.")
        self.neighbor.add(other.qubit_id)
        other.neighbor.add(self.qubit_id)

        return other

    def set_color(self, color: str) -> None:
        """
        Set the color of this qubit.

        Args:
            color (str): The new color.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> q0 = QubitNode(1)
            >>> q0.set_color('#ababab')
            >>> q0.color
            '#ababab'
        """
        _check_input_type("color", str, color)
        self.color_ = color

    def set_poi(self, poi_x: float, poi_y: float) -> None:
        """
        Set the position of this qubit.

        Args:
            poi_x (float): x position of qubit in canvas.
            poi_y (float): y position of qubit in canvas.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> q0 = QubitNode(1, poi_x=0, poi_y=1)
            >>> q0.set_poi(1, 0)
            >>> print(q0.poi_x, q0.poi_y)
            1 0
        """
        _check_input_type("poi_x", numbers.Real, poi_x)
        _check_input_type("poi_y", numbers.Real, poi_y)
        self.poi_x_, self.poi_y_ = poi_x, poi_y

    @property
    def color(self) -> str:
        """
        Get the color of this qubit.

        Returns:
            str, the color of qubit node.
        """
        return self.color_

    @property
    def qubit_id(self) -> int:
        """
        Get the id of this qubit.

        Returns:
            int, the id of this qubit.
        """
        return self.id_

    @property
    def poi_x(self) -> float:
        """
        X position of this qubit.

        Returns:
            float, the x position.
        """
        return self.poi_x_

    @property
    def poi_y(self) -> float:
        """
        Y position of this qubit.

        Returns:
            float, the y position.
        """
        return self.poi_y_


class QubitsTopology:
    """
    Topology of qubit in physical device.

    Topology is construct by different :class:`~.device.QubitNode`, and you
    can set the property of each qubit node directly.

    Args:
        qubits (List[:class:`~.device.QubitNode`]): All qubit nodes in this topology.

    Examples:
        >>> from mindquantum.device import QubitsTopology
        >>> from mindquantum.device import QubitNode
        >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
        >>> topology[0] >> topology[1]
        >>> topology.is_coupled_with(0, 1)
        True
        >>> topology.set_color(0, "#121212")
        >>> topology[0].color
        '#121212'
    """

    def __init__(self, qubits: typing.List[QubitNode]) -> None:
        """Initialize a physical qubit topology."""
        _check_input_type("qubits", list, qubits)
        for qubit in qubits:
            _check_input_type("Element of qubits", QubitNode, qubit)
        self.qubits: typing.Dict[int, QubitNode] = {}
        for node in qubits:
            if node.qubit_id in self.qubits:
                raise ValueError(f"Qubit with id {node.qubit_id} already exists.")
            self.qubits[node.qubit_id] = node
        self.cached_edge_color: typing.Dict[typing.Tuple[int, int], str] = {}

    def __getitem__(self, qubit_id: int) -> QubitNode:
        """
        Get qubit node base on qubit id.

        Args:
            qubit_id (int), the identity number of qubit you want.

        Returns:
            :class:`~.device.QubitNode`, the qubit node with given id.

        Examples:
            >>> from mindquantum.device import QubitsTopology
            >>> from mindquantum.device import QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[2].qubit_id
            2
        """
        _check_int_type("qubit_id", qubit_id)
        return self.qubits[qubit_id]

    def __get_cpp_obj__(self):
        """Get cpp object."""
        return QubitsTopology_([node.__get_cpp_obj__() for node in self.qubits.values()])

    def add_qubit_node(self, qubit: QubitNode) -> None:
        """
        Add a qubit node into this topology.

        Args:
            qubit (:class:`~.device.QubitNode`): the qubit you want to add into this topology.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.add_qubit_node(QubitNode(2));
            >>> topology.all_qubit_id()
            {0, 1, 2}
        """
        _check_input_type("qubit", QubitNode, qubit)
        if qubit.qubit_id in self.qubits:
            raise ValueError(f"Qubit with id {qubit.qubit_id} already exists.")
        self.qubits[qubit.qubit_id] = qubit

    def all_qubit_id(self) -> typing.Set[int]:
        """
        Get all qubit id.

        Returns:
            Set[int], all qubit id in this qubit topology.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.add_qubit_node(QubitNode(2));
            >>> topology.all_qubit_id()
            {0, 1, 2}
        """
        return set(self.qubits.keys())

    def compress(self) -> typing.Tuple["QubitsTopology", typing.Dict[int, int]]:
        """
        Relabeling the qubit id so that the qubit id in new topology will start from 0.

        Returns:
            Tuple[QubitsTopology, Dict[int, int]], the first element of return is the new compressed topology,
            and the second element of return is the qubit id map with key be the qubit id in old topology and
            value be the qubit id in new topology.

        Examples:
            >>> from mindquantum.device import LinearQubits
            >>> topo1 = LinearQubits(5)
            >>> topo1.remove_qubit_node(0)
            >>> topo1.remove_qubit_node(2)
            >>> topo2, qubit_map = topo1.compress()
            >>> print(topo2.edges_with_id())
            {(1, 2)}
            >>> print(qubit_map)
            {1: 0, 3: 1, 4: 2}
        """
        topo_dup = copy.deepcopy(self)
        old_id = sorted(self.all_qubit_id())
        nodes = topo_dup.choose(old_id)
        old_new_id_dict = {i: j for j, i in enumerate(old_id)}
        for qid in old_id:
            topo_dup.isolate_with_near(qid)
        for node in nodes:
            node.id_ = old_new_id_dict[node.id_]
        out = QubitsTopology(nodes)
        for i, j in self.edges_with_id():
            _ = out[old_new_id_dict[i]] >> out[old_new_id_dict[j]]
            edge_color = self.get_edge_color(i, j)
            if edge_color is not None:
                out.set_edge_color(old_new_id_dict[i], old_new_id_dict[j], edge_color)
        return out, old_new_id_dict

    def edges_with_id(self) -> typing.Set[typing.Tuple[int, int]]:
        """
        Get edges with id of two connected qubits.

        Returns:
            Set[Tuple[int, int]], all connected edges in this qubit topology.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology[0] << topology[1]
            >>> topology.edges_with_id()
            {(0, 1)}
        """
        out: typing.Set[typing.Tuple[int, int]] = set()
        for qubit1 in self.qubits.values():
            for qid_2 in qubit1.neighbor:
                if qubit1.qubit_id > qid_2:
                    out.add((qid_2, qubit1.qubit_id))
                else:
                    out.add((qubit1.qubit_id, qid_2))
        return out

    def edges_with_poi(self) -> typing.Set[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]:
        """
        Get edges with position of two connected qubits.

        Returns:
            Set[Tuple[Tuple[float, float], Tuple[float, float]]], the x and y position of two connected qubits.

        Examples:
            >>> from mindquantum.device import QubitNode, QubitsTopology
            >>> q0 = QubitNode(0, poi_x=0, poi_y=0)
            >>> q1 = QubitNode(1, poi_x=1, poi_y=0)
            >>> q0 >> q1
            >>> topology = QubitsTopology([q0, q1])
            >>> topology.edges_with_poi()
            {((0, 0), (1, 0))}
        """
        out: typing.Set[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]] = set()
        for q_id_1, q_id_2 in self.edges_with_id():
            q1 = self.qubits[q_id_1]
            q2 = self.qubits[q_id_2]
            out.add(((q1.poi_x, q1.poi_y), (q2.poi_x, q2.poi_y)))
        return out

    def has_qubit_node(self, qubit_id: int) -> bool:
        """
        Check whether a qubit is in this topology.

        Args:
            qubit_id (int): the id of qubit you want to check.

        Returns:
            bool, whether this topology has qubit node with given id.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.has_qubit_node(0)
            True
        """
        _check_int_type("qubit_id", qubit_id)
        return qubit_id in self.qubits

    def is_coupled_with(self, id1: int, id2: int) -> bool:
        """
        Check whether two qubit nodes are coupled.

        Args:
            id1 (int): the id of one qubit you want to check.
            id2 (int): the id of the other qubit you want to check.

        Returns:
            bool, whether two qubits node with given ids coupled.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.is_coupled_with(0, 1)
            False
            >>> topology[0] >> topology[1]
            >>> topology.is_coupled_with(0, 1)
            True
        """
        _check_int_type("id1", id1)
        _check_int_type("id2", id2)
        return id2 in self.qubits[id1].neighbor

    def isolate_with_near(self, qubit_id: int) -> None:
        """
        Disconnect with all coupling qubits.

        Args:
            qubit_id (int): the id of qubit you want to disconnect with all nearby qubits.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[0] >> topology[1] >> topology[2]
            >>> topology.edges_with_id()
            {(0, 1), (1, 2)}
            >>> topology.isolate_with_near(1)
            >>> topology.edges_with_id()
            set()
        """
        _check_int_type("qubit_id", qubit_id)
        current_node = self[qubit_id]
        other_nodes = [self[i] for i in current_node.neighbor]
        for node in other_nodes:
            _ = node > current_node

    def n_edges(self) -> int:
        """
        Get total connected edge number.

        Returns:
            int, the edge number of this qubit topology.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[0] >> topology[1] >> topology[2]
            >>> topology.n_edges()
            2
        """
        return sum(len(i.neighbor) for i in self.qubits.values()) >> 1

    def choose(self, ids: typing.List[int]) -> typing.List[QubitNode]:
        """
        Choose qubit nodes based on given qubit id.

        Args:
            ids (List[int]): A list of qubit node id.

        Returns:
            List[:class:`~.device.QubitNode`], a list of qubit node based on given qubit node id.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[0] >> topology[1] >> topology[2]
            >>> nodes = topology.choose([0, 1])
            >>> print(nodes[0].qubit_id, nodes[1].qubit_id)
            0 1
        """
        _check_input_type("ids", list, ids)
        for qubit_id in ids:
            _check_int_type("Element of ids", qubit_id)
        return [self[i] for i in ids]

    def remove_isolate_node(self) -> None:
        """
        Remove qubit node that do not connect with any other qubits.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[0] >> topology[1] >> topology[2]
            >>> topology.edges_with_id()
            {(0, 1), (1, 2)}
            >>> topology.isolate_with_near(1)
            >>> topology.all_qubit_id()
            {0, 1, 2}
            >>> topology.remove_isolate_node()
            >>> topology.all_qubit_id()
            set()
        """
        will_remove = []
        for node in self.qubits.values():
            if not node.neighbor:
                will_remove.append(node.qubit_id)

        for qubit_id in will_remove:
            _ = self.qubits.pop(qubit_id)

    def remove_qubit_node(self, qubit_id: int) -> None:
        """
        Remove a qubit node out of this topology.

        Args:
            qubit_id (int): the id of qubit you want to remove.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.remove_qubit_node(1)
            >>> topology.all_qubit_id()
            {0, 2}
        """
        _check_int_type("qubit_id", qubit_id)
        will_remove = self.qubits[qubit_id]
        near_qubits = [self[i] for i in will_remove.neighbor]
        for node in near_qubits:
            _ = node > will_remove
        self.qubits.pop(qubit_id)

    def select(self, ids: typing.List[int]) -> "QubitsTopology":
        """
        Select certain qubit nodes to generate a new topology.

        Args:
            ids (List[int]): A list of qubit node id.

        Returns:
            :class:`~.device.QubitsTopology`, a new topology while keeping the connection property.

        Examples:
            >>> from mindquantum.device import LinearQubits
            >>> t1 = LinearQubits(4)
            >>> t2 = t1.select([0, 1, 2])
            >>> t2.edges_with_id()
            {(0, 1), (1, 2)}
        """
        ids_set = set(ids)
        nodes = copy.deepcopy(self.choose(ids))
        for node in nodes:
            node.neighbor &= ids_set
        out = QubitsTopology(nodes)
        for (i, j), c in self.cached_edge_color.items():
            if out.is_coupled_with(i, j):
                out.set_edge_color(i, j, c)
        return out

    def set_color(self, qubit_id: int, color: str) -> None:
        """
        Set color of certain qubit.

        Args:
            qubit_id (int): the id of qubit you want to change color.
            color (str): the new color.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.set_color(0, "#ababab")
            >>> topology[0].color
            '#ababab'
        """
        _check_int_type("qubit_id", qubit_id)
        _check_input_type("color", str, color)
        self[qubit_id].color_ = color

    def set_edge_color(self, qubit_id1: int, qubit_id2: int, color: str) -> None:
        """
        Set color of edge.

        The order of qubit_id1 and qubit_id2 does not matter.

        Args:
            qubit_id1 (int): The first qubit of edge.
            qubit_id2 (int): The second qubit of edge.
            color (str): The color of edge.
        """
        _check_input_type("color", str, color)
        if not self.is_coupled_with(qubit_id1, qubit_id2):
            raise ValueError(f"qubit {qubit_id1} is not connected with qubit {qubit_id2}")
        self.cached_edge_color[tuple(sorted([qubit_id1, qubit_id2]))] = color

    def get_edge_color(self, qubit_id1: int, qubit_id2: int) -> str:
        """
        Get color of edge.

        The order of qubit_id1 and qubit_id2 does not matter.

        Args:
            qubit_id1 (int): The first qubit of edge.
            qubit_id2 (int): The second qubit of edge.
        """
        return self.cached_edge_color.get(tuple(sorted([qubit_id1, qubit_id2])), None)

    def set_position(self, qubit_id: int, poi_x: float, poi_y: float) -> None:
        """
        Set position of a certain qubit.

        Args:
            qubit_id (int): the id of qubit you want to change position.
            poi_x (float): new x position.
            poi_y (float): new y position.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.set_position(0, 1, 1)
            >>> topology[0].poi_x, topology[0].poi_y
            (1, 1)
        """
        _check_int_type("qubit_id", qubit_id)
        _check_input_type("poi_x", numbers.Real, poi_x)
        _check_input_type("poi_y", numbers.Real, poi_y)
        self[qubit_id].set_poi(poi_x, poi_y)

    def show(self, method: Optional[AVA_SHOW_METHOD] = None):
        """
        Display the topology.

        Args:
            method (str): The method you want to display the topology. If ``None``, we will use default method,
                which is ``'mpl'`` in terminal environment and ``'svg'`` in jupyter notebook environment. You
                can also set it to ``'mpl'`` or ``'svg'`` manually. Default: ``None``.
        """
        # pylint: disable=import-outside-toplevel
        from mindquantum.io.display import draw_topology, draw_topology_plt

        is_jupyter = Console().is_jupyter
        if method is None:
            method = 'svg' if is_jupyter else 'mpl'
        if method not in get_args(AVA_SHOW_METHOD):
            raise ValueError(f"method should be one of {get_args(AVA_SHOW_METHOD)}, but get {method}.")
        if method == 'svg':
            return draw_topology(self)
        return draw_topology_plt(self)

    def size(self) -> int:
        """
        Get total qubit number.

        Returns:
            int, the total qubit number.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.size()
            3
        """
        return len(self.qubits)


class LinearQubits(QubitsTopology):
    """
    Linear qubit topology.

    Args:
        n_qubits (int): total qubit size.

    Examples:
        >>> from mindquantum.device import LinearQubits
        >>> topology = LinearQubits(5)
        >>> topology.is_coupled_with(0, 1)
        True
        >>> topology.is_coupled_with(0, 2)
        False
    """

    def __init__(self, n_qubits: int):
        """Initialize a linear topology."""
        _check_int_type("n_qubits", n_qubits)
        nodes = [QubitNode(i, poi_x=i) for i in range(n_qubits)]
        left_node = nodes[0]
        for node in nodes[1:]:
            left_node = left_node >> node
        super().__init__(nodes)


class GridQubits(QubitsTopology):
    """
    Grid qubit topology.

    Args:
        n_row (int): how many rows of your grid qubits.
        n_col (int): how many columns of your grid quits.

    Examples:
        >>> from mindquantum.device import GridQubits
        >>> topology = GridQubits(2, 3)
        >>> topology.n_row()
        2
    """

    def __init__(self, n_row: int, n_col: int) -> None:
        """Initialize a grid topology with row and col number."""
        _check_int_type("n_row", n_row)
        _check_int_type("n_col", n_col)
        self.n_row_ = n_row
        self.n_col_ = n_col
        qubits = []
        for r in range(n_row):
            for c in range(n_col):
                qubits.append(QubitNode(r * n_col + c, poi_x=c, poi_y=r))
        for r in range(n_row):
            next_node = qubits[r * n_col]
            for c in range(n_col - 1):
                next_node = next_node >> qubits[next_node.qubit_id + 1]
        for c in range(n_col):
            next_node = qubits[c]
            for r in range(n_row - 1):
                next_node = next_node >> qubits[next_node.qubit_id + n_col]
        super().__init__(qubits)

    def n_col(self) -> int:
        """
        Get column number.

        Returns:
            int, the column number.

        Examples:
            >>> from mindquantum.device import GridQubits
            >>> topology = GridQubits(2, 3)
            >>> topology.n_col()
            3
        """
        return self.n_col_

    def n_row(self) -> int:
        """
        Get row number.

        Returns:
            int, the row number.

        Examples:
            >>> from mindquantum.device import GridQubits
            >>> topology = GridQubits(2, 3)
            >>> topology.n_row()
            2
        """
        return self.n_row_
